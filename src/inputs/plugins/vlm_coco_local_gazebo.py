# type: ignore

import asyncio
import collections
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from google.protobuf import text_format
from om1_vlm.gz.msgs import image_pb2  # noqa
from PIL import Image
from torchvision.models import detection as detection_model

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

Detection = collections.namedtuple("Detection", "label, bbox, score")


@dataclass
class Message:
    """
    Container for timestamped messages.

    Parameters
    ----------
    timestamp : float
        Unix timestamp of the message
    message : str
        Content of the message
    """

    timestamp: float
    message: str


# if working on Mac, please disable continuity camera on your iphone
# Settings > General > AirPlay & Continuity, and tunr off Continuity


class VLM_COCO_Local_Gazebo(FuserInput[Image.Image]):
    """
    Detects COCO objects in image and publishes messages.
    Uses PyTorch and FasterRCNN_MobileNet model from torchvision.
    Bounding Boxes use image convention, ie center.y = 0 means top of image.
    """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize VLM input handler with empty message buffer.
        """
        super().__init__(config)

        self.device = "cpu"
        self.detection_threshold = 0.3

        # Track IO
        self.io_provider = IOProvider()

        # Messages buffer
        self.messages: list[Message] = []

        # Simple description of sensor output to help LLM understand its importance and utility
        self.descriptor_for_LLM = "Object Detector"

        # Low resolution Faster R-CNN model with a MobileNetV3-Large backbone tuned for mobile use cases.
        self.model = detection_model.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1",
            progress=True,
            weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        ).to(self.device)

        self.class_labels = (
            detection_model.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta[
                "categories"
            ]
        )
        self.model.eval()
        logging.info("COCO Object Detector Started")

        self.cam_third = 0  # This will be updated for _process_image is called

        self.topic = getattr(self.config, "topic", "/camera")

    def _parse_text_message(self, text_data):
        """
        Parses a text-formatted protobuf message (with escaped binary data)
        into an Image message.
        """
        img_msg = image_pb2.Image()
        try:
            # The text_format parser automatically unescapes bytes fields.
            text_format.Parse(text_data, img_msg)
        except Exception as e:
            logging.error("Text parsing failed:", e)
            return None
        return img_msg

    def _process_image(self, img_msg):
        """
        Converts the protobuf Image message into a NumPy array, taking into account
        the row stride (step) and converting color channels for OpenCV.
        """
        width = img_msg.width
        self.cam_third = int(width / 3)
        height = img_msg.height
        step = img_msg.step  # Number of bytes per row (may include padding)
        encoding = img_msg.pixel_format_type
        raw_data = img_msg.data  # The text parser converts the escaped string to bytes

        # Determine number of channels based on pixel format.
        if encoding == image_pb2.PixelFormatType.RGB_INT8:
            channels = 3
        elif encoding == image_pb2.PixelFormatType.BGR_INT8:
            channels = 3
        elif encoding == image_pb2.PixelFormatType.RGBA_INT8:
            channels = 4
        elif encoding == image_pb2.PixelFormatType.GRAY8:
            channels = 1
        else:
            logging.error(f"Unsupported pixel format: {encoding}")
            return None

        expected_bytes_per_row = width * channels
        if step < expected_bytes_per_row:
            logging.error(
                f"Step value ({step}) is less than expected row bytes ({expected_bytes_per_row})."
            )
            return None

        try:
            if step == expected_bytes_per_row:
                image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                    (height, width, channels)
                )
            else:
                # Reshape to (height, step) then crop each row to the actual image data.
                rows = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, step))
                image_array = rows[:, :expected_bytes_per_row].reshape(
                    (height, width, channels)
                )
        except Exception as e:
            logging.error("Error reshaping image data:", e)
            return None

        # Convert color channels as needed. OpenCV expects BGR.
        if encoding == image_pb2.PixelFormatType.RGB_INT8:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif encoding == image_pb2.PixelFormatType.RGBA_INT8:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        elif encoding == image_pb2.PixelFormatType.GRAY8:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        # BGR_INT8: no conversion needed

        return image_array

    def _get_message(self):
        """
        Runs gz topic to fetch one message from /camera in text format.
        """
        try:
            result = subprocess.run(
                ["gz", "topic", "-e", "-t", self.topic, "-n", "1"],
                capture_output=True,
                text=True,
                timeout=5,  # Avoid indefinite hangs
            )

            if result.returncode != 0:
                logging.error(f"Command failed with error: {result.stderr}")
                return None

            return result.stdout

        except subprocess.TimeoutExpired:
            logging.error(
                f"Subprocess timed out while fetching message from {self.topic}."
            )
            return None

        except FileNotFoundError:
            logging.error(
                "The 'gz' command was not found. Ensure that Gazebo is installed and in PATH."
            )
            return None

        except Exception as e:
            logging.error(f"Unexpected error in _get_message: {e}")
            return None

    async def _poll(self) -> Image.Image:
        """
        Poll for new image input.

        Currently generates random colored images for testing.
        In production, this would interface with camera or sensor.

        Returns
        -------
        Image.Image
            Generated or captured image
        """
        await asyncio.sleep(0.5)

        text_data = self._get_message()
        if not text_data:
            return

        # Parse the text-formatted protobuf message.
        img_msg = self._parse_text_message(text_data)
        if img_msg is None:
            logging.debug("Unable to parse current image message")
            return
        # Process the image from the parsed message.
        image = self._process_image(img_msg)
        if image is None:
            logging.debug("Unable to process current image message")
            return

        return image

    async def _raw_to_text(self, raw_input: Optional[Image.Image]) -> Optional[Message]:
        """
        Process raw image input to generate text description.

        Parameters
        ----------
        raw_input : Image.Image
            Input image to process

        Returns
        -------
        Message
            Timestamped message containing description
        """

        filtered_detections = None

        full_detections = None

        if raw_input is not None:
            image = raw_input.copy().transpose((2, 0, 1))
            batch_image = np.expand_dims(image, axis=0)
            tensor_image = torch.tensor(
                batch_image / 255.0, dtype=torch.float, device=self.device
            )
            mobilenet_detections = self.model(tensor_image)[
                0
            ]  # pylint: disable=E1102 disable not callable warning
            filtered_detections = [
                Detection(label_id, box, score)
                for label_id, box, score in zip(
                    mobilenet_detections["labels"],
                    mobilenet_detections["boxes"],
                    mobilenet_detections["scores"],
                )
                if score >= self.detection_threshold
            ]
            full_detections = [
                Detection(label_id, box, score)
                for label_id, box, score in zip(
                    mobilenet_detections["labels"],
                    mobilenet_detections["boxes"],
                    mobilenet_detections["scores"],
                )
            ]
            logging.debug(f"COCO filtered_detections {filtered_detections}")

        sentence = None

        if filtered_detections and len(filtered_detections) > 0:

            pred_boxes = torch.stack(
                [detection.bbox for detection in filtered_detections]
            )
            pred_scores = torch.stack(
                [detection.score for detection in filtered_detections]
            )
            pred_labels = [
                self.class_labels[detection.label] for detection in filtered_detections
            ]
            logging.debug(f"COCO labels {pred_labels} scores {pred_scores}")

            # we have a least one detection, and that will have the highest score
            thing = pred_labels[0]
            x1 = pred_boxes[0, 0]
            x2 = pred_boxes[0, 2]
            center_x = (x1 + x2) / 2  # center of the bbox

            direction = "in front of you"

            if center_x < self.cam_third:
                direction = "on your left"
            elif center_x > 2 * self.cam_third:
                direction = "on your right"

            sentence = f"You see a {thing} {direction}."

            # add at most one more object
            if len(pred_labels) > 1:
                other_thing = pred_labels[1]
                sentence = sentence + f" You also see a {other_thing}."

        elif full_detections and len(full_detections) > 0:
            full_labels = [
                self.class_labels[detection.label] for detection in full_detections
            ]
            logging.info(
                f"COCO isn't detecting anything familiar. The closest thing it recognises is {full_labels[0]}"
            )
        else:
            logging.info("COCO isn't detecting anything")

        if sentence is not None:
            return Message(timestamp=time.time(), message=sentence)

    async def raw_to_text(self, raw_input: Image.Image):
        """
        Convert raw image to text and update message buffer.

        Parameters
        ----------
        raw_input : Image.Image
            Raw image to be processed
        """
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Formats the most recent message with timestamp and class name,
        adds it to the IO provider, then clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        logging.info(f"VLM_COCO_Local_Gazebo: {latest_message.message}")

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
