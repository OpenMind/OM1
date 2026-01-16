import asyncio
import collections
import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
from pydantic import Field
from torchvision.models import detection as detection_model

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class VLM_COCO_LocalConfig(SensorConfig):
    """
    Configuration for Local COCO VLM Sensor.

    Parameters
    ----------
    camera_index : int
        Index of the camera device.
    """

    camera_index: int = Field(default=0, description="Index of the camera device")


Detection = collections.namedtuple("Detection", "label, bbox, score")

# if working on Mac, please disable continuity camera on your iphone
# Settings > General > AirPlay & Continuity, and tunr off Continuity


def check_webcam(index_to_check):
    """
    Checks if a webcam is available and returns True if found, False otherwise.
    """
    cap = cv2.VideoCapture(index_to_check)  # 0 is the default camera index
    if not cap.isOpened():
        logging.info(f"ERROR: COCO did not find cam: {index_to_check}")
        return False
    logging.info(f"COCO found cam: {index_to_check}")
    return True


class VLM_COCO_Local(FuserInput[VLM_COCO_LocalConfig, Optional[np.ndarray]]):
    """
    Detects COCO objects in image and publishes messages.
    Uses PyTorch and FasterRCNN_MobileNet model from torchvision.
    Bounding Boxes use image convention, ie center.y = 0 means top of image.
    """

    def __init__(self, config: VLM_COCO_LocalConfig):
        """
        Initialize VLM input handler with empty message buffer.
        """
        super().__init__(config)

        self.device = "cpu"
        self.detection_threshold = 0.2

        self.camera_index = self.config.camera_index

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

        self.have_cam = check_webcam(self.camera_index)

        # Start capturing video, if we have a webcam
        self.cap = None
        if self.have_cam:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.width = int(self.cap.get(3))  # float `width`
            self.height = int(self.cap.get(4))  # float `height`
            self.cam_third = int(self.width / 3)
            logging.info(
                f"Webcam pixel dimensions for COCO: {self.width}, {self.height}"
            )

    async def _poll(self) -> Optional[np.ndarray]:
        """
        Poll for new image input.

        Currently generates random colored images for testing.
        In production, this would interface with camera or sensor.

        Returns
        -------
        Optional[np.ndarray]
            Generated or captured image as a numpy array
        """
        await asyncio.sleep(0.5)

        # Capture a frame every 500 ms
        # logging.info(f"VLM_COCO_Local poll")

        if self.have_cam and self.cap is not None:
            ret, frame = self.cap.read()
            # logging.info(f"VLM_COCO_Local frame: {frame}")
            return frame

        return None

    async def _raw_to_text(self, raw_input: Optional[np.ndarray]) -> Optional[Message]:
        """
        Process raw image input to generate text description.

        Parameters
        ----------
        raw_input : np.ndarray
            Input numpy array image to process

        Returns
        -------
        Message
            Timestamped message containing description
        """
        filtered_detections = None

        if raw_input is not None:
            image = raw_input.copy().transpose((2, 0, 1))

            batch_image = np.expand_dims(image, axis=0)
            tensor_image = torch.tensor(
                batch_image / 255.0, dtype=torch.float, device=self.device
            )
            mobilenet_detections = self.model(tensor_image)[
                0
            ]  # pylint: disable=E1102 disable not callable warning

            # logging.info(f"VLM_COCO_Local detections: {mobilenet_detections}")
            filtered_detections = [
                Detection(label_id, box, score)
                for label_id, box, score in zip(
                    mobilenet_detections["labels"],
                    mobilenet_detections["boxes"],
                    mobilenet_detections["scores"],
                )
                if score >= self.detection_threshold
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
            # so if the width is 1920, then the left third runs between 0 and 639
            # middle is 640 - 1279
            # right is > 1280
            if center_x < self.cam_third:
                direction = "on your left"
            elif center_x > 2 * self.cam_third:
                direction = "on your right"

            sentence = f"You see a {thing} {direction}."

            # add at most one more object
            if len(pred_labels) > 1:
                other_thing = pred_labels[1]
                sentence = sentence + f" You also see a {other_thing}."

        if sentence is not None:
            return Message(timestamp=time.time(), message=sentence)

    async def raw_to_text(self, raw_input: Optional[np.ndarray]):
        """
        Convert raw image to text and update message buffer.

        Parameters
        ----------
        raw_input : Optional[np.ndarray]
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

        logging.info(f"VLM_COCO_Local: {latest_message.message}")

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
