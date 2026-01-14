import base64
import logging
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from mjpeg.client import MJPEGClient
from om1_vlm import VideoStream

from ubtech.ubtechapi import YanAPI


class UbtechCameraVideoStream(VideoStream):
    """
    Video stream handler for Ubtech robots using YanAPI and MJPEGClient.
    """

    def __init__(
        self,
        robot_ip: str,
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        fps: Optional[int] = 30,
        resolution: Optional[Tuple[int, int]] = (640, 480),
        jpeg_quality: int = 70,
    ):
        super().__init__(
            frame_callback=frame_callback,
            frame_callbacks=frame_callbacks,
            fps=fps,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
        )

        self.robot_ip = robot_ip
        self.url = f"http://{self.robot_ip}:8000/stream.mjpg"
        self.stream_client: Optional[MJPEGClient] = None

        YanAPI.yan_api_init(self.robot_ip)

    def on_video(self):
        """
        Main loop to handle video streaming from the Ubtech robot.
        """
        logging.info("Starting Ubtech MJPEG video stream")

        try:
            self.resolution = self.resolution or (640, 480)
            YanAPI.open_vision_stream(
                resolution=f"{self.resolution[0]}x{self.resolution[1]}"
            )
            time.sleep(2)

            self.stream_client = MJPEGClient(self.url)
            bufs = self.stream_client.request_buffers(65536, 50)
            for b in bufs:
                self.stream_client.enqueue_buffer(b)
            self.stream_client.start()

            frame_time = 1.0 / (self.fps or 30)
            last_time = time.perf_counter()

            while self.running:
                try:
                    buf = self.stream_client.dequeue_buffer()
                    frame_bytes = np.frombuffer(buf.data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
                    self.stream_client.enqueue_buffer(buf)

                    if frame is not None:
                        height, width = frame.shape[:2]
                        ratio = width / height
                        new_width, new_height = (
                            (self.resolution[0], int(self.resolution[0] / ratio))
                            if width > height
                            else (int(self.resolution[1] * ratio), self.resolution[1])
                        )
                        resized = cv2.resize(
                            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
                        )
                        _, buffer = cv2.imencode(".jpg", resized, self.encode_quality)
                        frame_data = base64.b64encode(buffer.tobytes()).decode("utf-8")

                        for cb in self.frame_callbacks:
                            cb(frame_data)
                    else:
                        logging.warning("Received empty frame")

                    elapsed = time.perf_counter() - last_time
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                    last_time = time.perf_counter()

                except Exception as e:
                    logging.error(f"Video processing error: {e}")
        finally:
            if self.stream_client:
                self.stream_client.stop()
                logging.info("Stopped MJPEG stream client")

            YanAPI.close_vision_stream()
            logging.info("Closed vision stream on robot")
