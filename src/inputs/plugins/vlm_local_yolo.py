import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.odom_provider import OdomProvider

# Common resolutions to test (width, height), ordered high to low
RESOLUTIONS = [
    (3840, 2160),  # 4K
    (2560, 1440),  # QHD
    (1920, 1080),  # Full HD
    (1280, 720),  # HD
    (1024, 576),
    (800, 600),
    (640, 480),  # VGA fallback
]


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


def set_best_resolution(cap, resolutions):
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Give it a moment to settle
        time.sleep(0.1)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width == width and actual_height == height:
            logging.info(f"✅ Resolution set to: {width}x{height}")
            return width, height

    logging.info("⚠️ Could not set preferred resolution. Using default.")
    return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


# if working on Mac, please disable continuity camera on your iphone
# Settings > General > AirPlay & Continuity, and turn off Continuity
def check_webcam(index_to_check):
    """
    Checks if a webcam is available and returns True if found, False otherwise.
    """
    cap = cv2.VideoCapture(index_to_check)
    if not cap.isOpened():
        logging.error(f"YOLO did not find cam: {index_to_check}")
        return 0, 0

    # Set the best available resolution
    width, height = set_best_resolution(cap, RESOLUTIONS)
    logging.info(f"YOLO found cam: {index_to_check} set to {width}{height}")
    return width, height


class VLM_Local_YOLO(FuserInput[str]):
    """ """

    def __init__(self, config: SensorConfig = SensorConfig()):
        """
        Initialize VLM input handler with empty message buffer.
        """
        super().__init__(config)

        self.camera_index = 0  # default to default webcam unless specified otherwise
        if self.config.camera_index:
            self.camera_index = self.config.camera_index

        # Track IO
        self.io_provider = IOProvider()

        # Messages buffer
        self.messages: list[Message] = []

        # Simple description of sensor output to help LLM understand its importance and utility
        self.descriptor_for_LLM = "Eyes"

        # Load model
        self.model = YOLO("yolo11n-seg.pt")

        self.write_to_local_file = False
        if self.config.log_file:
            self.write_to_local_file = self.config.log_file

        self.filename_current = None
        self.max_file_size_bytes = 1024 * 1024

        # Create timestamped log filename
        if self.write_to_local_file:
            self.filename_current = self.update_filename()
            logging.info(f"YOLO Logging to {self.filename_current}")

        self.width, self.height = check_webcam(self.camera_index)

        self.have_cam = False

        if self.width > 0:
            self.have_cam = True

        self.frame_index = 0

        # Start capturing video, if we have a webcam
        self.cap = None
        if self.have_cam:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cam_third = int(self.width / 3)
            logging.info(
                f"Webcam pixel dimensions for YOLO: {self.width}, {self.height}"
            )

        self.odom = OdomProvider()
        logging.info(f"YOLO Odom Provider: {self.odom}")
        self.odom_rockchip_ts = 0.0
        self.odom_subscriber_ts = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw_0_360 = 0.0
        self.odom_yaw_m180_p180 = 0.0

    def mask_to_polygons(self, mask):
        """
        Convert a binary mask to polygons using OpenCV.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
        return polygons

    def update_filename(self):
        unix_ts = round(time.time(), 6)
        logging.info(f"YOLO time: {unix_ts}")
        unix_ts = str(unix_ts).replace(".", "_")
        filename = f"dump/yolo_{unix_ts}Z.jsonl"
        return filename

    def update_filename_img(self, frame_idx):
        unix_ts = round(time.time(), 6)
        # logging.info(f"IMG time: {unix_ts}")
        unix_ts = str(unix_ts).replace(".", "_")
        filename = f"dump/img_{frame_idx}_{unix_ts}Z.jpg"
        return filename

    def get_top_detection(self, detections):
        """
        Returns the class label and bbox of the detection with the highest confidence.

        Parameters:
            detections (list): List of detection dictionaries, each with 'class', 'confidence', 'bbox'.

        Returns:
            tuple: (label, bbox) of the top detection, or (None, None) if list is empty.
        """
        if not detections:
            return None, None

        top = max(detections, key=lambda d: d["confidence"])
        return top["class"], top["bbox"]

    async def _poll(self) -> Optional[List]:
        """
        Poll for new image input.

        Returns
        -------
        np.ndarray
            Generated or captured image as a numpy array
        """
        await asyncio.sleep(0.25)

        if self.have_cam:

            try:
                o = self.odom.position
                logging.debug(f"Odom data: {o}")
                if o:
                    self.odom_x = o["odom_x"]
                    self.odom_y = o["odom_y"]
                    self.odom_rockchip_ts = o["odom_rockchip_ts"]
                    self.odom_subscriber_ts = o["odom_subscriber_ts"]
                    self.odom_yaw_0_360 = o["odom_yaw_0_360"]
                    self.odom_yaw_m180_p180 = o["odom_yaw_m180_p180"]
            except Exception as e:
                logging.error(f"Error parsing Odom: {e}")

            ret, frame = self.cap.read()
            self.frame_index += 1

            fn = self.update_filename_img(self.frame_index)
            # logging.info(f"filename {fn}")
            try:
                if len(frame) > 0 and self.write_to_local_file:
                    cv2.imwrite(fn, frame)
                    # logging.info(f"Image saved successfully as {fn}")
                else:
                    logging.error("Error: Could not save frame from webcam.")
            except Exception as e:
                logging.error(f"Error saving IMG: {str(e)}")

            timestamp = time.time()
            results = self.model.predict(source=frame)

            data = []
            for result in results:
                frame_entry = {}

                if result.masks is None or result.boxes is None:
                    continue

                masks = result.masks.data.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                labels = (
                    [result.names[i] for i in class_ids]
                    if hasattr(result, "names")
                    else [str(i) for i in class_ids]
                )
                bboxes = result.boxes.xyxy.cpu().numpy().tolist()

                frame_entry["bboxes"] = bboxes
                frame_entry["labels"] = labels
                frame_entry["polygons"] = []

                for mask in masks:
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    polygons = self.mask_to_polygons(binary_mask)
                    # Convert each polygon to a single-line string
                    flat_polygons = [
                        "[" + ",".join(map(str, poly)) + "]" for poly in polygons
                    ]
                    frame_entry["polygons"].append(flat_polygons)

                data.append(frame_entry)

            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls]
                    detections.append(
                        {
                            "class": label,
                            "confidence": round(conf, 4),
                            "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        }
                    )

            # logging.info(
            #     f"\nFrame {self.frame_index} @ {timestamp} — {len(detections)} objects"
            # )

            if self.write_to_local_file:
                try:
                    json_line = json.dumps(
                        {
                            "frame": self.frame_index,
                            "timestamp": timestamp,
                            "data": data,
                            "odom_rockchip_ts": self.odom_rockchip_ts,
                            "odom_subscriber_ts": self.odom_subscriber_ts,
                            "odom_x": self.odom_x,
                            "odom_y": self.odom_y,
                            "odom_yaw_0_360": self.odom_yaw_0_360,
                            "odom_yaw_m180_p180": self.odom_yaw_m180_p180,
                        }
                    )
                    self.write_str_to_file(json_line)
                    logging.info(f"YOLO wrote to {self.filename_current}")
                except Exception as e:
                    logging.error(f"Error saving YOLO: {str(e)}")

            return detections

    def write_str_to_file(self, json_line: str):
        """
        Writes a dictionary to a file in JSON lines format. If the file exceeds max_file_size_bytes,
        creates a new file with a timestamp.

        Parameters:
        - data: Dictionary to write
        """

        if not isinstance(json_line, str):
            raise ValueError("Provided json_line must be a json string.")

        if (
            os.path.exists(self.filename_current)
            and os.path.getsize(self.filename_current) > self.max_file_size_bytes
        ):
            self.filename_current = self.update_filename()
            logging.info(f"New yolo file name: {self.filename_current}")

        with open(self.filename_current, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")
            f.flush()

    async def _raw_to_text(self, raw_input: Optional[List]) -> Optional[Message]:
        """
        Process raw image input to generate text description.

        Parameters
        ----------
        raw_input : List of YOLO detections

        Returns
        -------
        Message
            Timestamped message containing description
        """

        detections = raw_input

        if detections:
            for det in detections:
                logging.debug(
                    f"{det['class']} ({det['confidence']:.2f}) -> {det['bbox']}"
                )

            sentence = None

            thing, bbox = self.get_top_detection(detections)
            x1 = bbox[0]
            x2 = bbox[2]
            center_x = (x1 + x2) / 2  # center of the bbox

            direction = "in front of you"
            if center_x < self.cam_third:
                direction = "on your left"
            elif center_x > 2 * self.cam_third:
                direction = "on your right"

            sentence = f"You see a {thing} {direction}."

            if sentence is not None:
                return Message(timestamp=time.time(), message=sentence)

    async def raw_to_text(self, raw_input: List):
        """
        Convert list of detections to text and update message buffer.

        Parameters
        ----------
        raw_input : List[detections]
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

        logging.info(f"VLM_YOLO_Local: {latest_message.message}")

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
