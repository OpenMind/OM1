import asyncio
import base64
import json
import logging
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml
from om1_vlm import VideoRTSPStream
from ultralytics import YOLO

from .io_provider import IOProvider
from .singleton import singleton

logger = logging.getLogger(__name__)


def now_ms() -> int:
    return int(time.time() * 1000)


DEFAULT_DETECT_CFG = {
    "detect": {
        "model": "yolov8n.pt",
        "conf": 0.35,
        "img_size": 640,
        "blacklist": ["person"],
        "whitelist": ["tv", "cell phone", "cup", "bottle", "book"],
    }
}


class Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.35,
        img_size: int = 640,
        blacklist=None,
        whitelist=None,
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.img_size = img_size
        self.blacklist = set(blacklist or [])
        self.whitelist = set(whitelist or [])

    def infer(self, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """
        Run YOLO on a single frame and return a list of detections:
        [{ "xyxy": [x1,y1,x2,y2], "label": str, "conf": float }, ...]
        """
        res = self.model.predict(
            source=frame_bgr, imgsz=self.img_size, conf=self.conf, verbose=False
        )[0]
        out: list[dict[str, Any]] = []
        boxes: Any = getattr(res, "boxes", None)
        if boxes is None:
            return out

        for b in boxes:
            cls_id = int(b.cls.item())
            label = res.names[cls_id]
            if label in self.blacklist:
                continue
            if self.whitelist and (label not in self.whitelist):
                continue
            xyxy = b.xyxy[0].tolist()
            conf = float(b.conf.item())
            out.append(dict(xyxy=xyxy, label=label, conf=conf))
        return out


@singleton
class YoloDetectRTSPProvider:
    """
    RTSP → YOLO detection provider.

    - Subscribes to VideoRTSPStream via a frame callback (base64 JSON).
    - Processes frames on a background queue with stride + drop-oldest.
    - Runs YOLO detection in a worker thread.
    - Publishes latest {ts, frame, detections} into IOProvider under
      a given dynamic variable key.
    """

    IO_KEY_LATEST = "yolo_latest"

    def __init__(
        self,
        cfg_path: str | None = None,
        rtsp_url: str = "rtsp://localhost:8554/top_camera",
        decode_format: str = "H264",
        fps: int = 10,
        ingest_stride: int = 10,
        queue_max: int = 10,
    ):
        self.running: bool = False

        # Load config
        if (
            cfg_path is not None
            and yaml
            and isinstance(cfg_path, str)
            and cfg_path
            and isinstance(cfg_path, str)
        ):
            try:
                self.cfg = yaml.safe_load(open(cfg_path, "r"))
            except Exception:
                logger.exception("Failed to load detect cfg_path; using defaults")
                self.cfg = DEFAULT_DETECT_CFG.copy()
        else:
            self.cfg = DEFAULT_DETECT_CFG.copy()

        detect_cfg = self.cfg["detect"]

        self.detector = Detector(
            detect_cfg["model"],
            detect_cfg["conf"],
            detect_cfg["img_size"],
            detect_cfg.get("blacklist"),
            detect_cfg.get("whitelist"),
        )

        # IOProvider for publishing detections
        self.io_provider = IOProvider()

        # RTSP stream
        self.video_stream: VideoRTSPStream = VideoRTSPStream(
            rtsp_url,
            decode_format,
            frame_callback=self._on_frame,
            fps=fps,
        )
        self._owns_stream = True

        # Ingest control
        self._ingest_stride = max(1, int(ingest_stride))
        self._frame_counter = 0
        self._task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(queue_max)))

    # -------------------- External API --------------------

    def register_frame_callback(self, video_callback: Optional[Callable[[str], None]]):
        if video_callback is not None:
            self.video_stream.register_frame_callback(video_callback)

    def start(self):
        if self.running:
            logger.warning("YoloDetectRTSPProvider is already running")
            return
        self.running = True

        if self._owns_stream:
            self.video_stream.start()

        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._consumer())
        logger.info("YoloDetectRTSPProvider started")

    def stop(self):
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()

        if self._owns_stream:
            try:
                self.video_stream.stop()
            except Exception:
                pass
        logger.info("YoloDetectRTSPProvider stopped")

    # -------------------- Callbacks & Worker --------------------

    def _on_frame(self, frame_data: str):
        """
        Receives base64 JSON from VideoRTSPStream, applies stride, and enqueues (drop-oldest).
        frame_data: JSON string {"timestamp": float, "frame": <base64 jpeg>}
        """
        try:
            self._frame_counter += 1
            if (self._frame_counter % self._ingest_stride) != 0 or not self.running:
                return

            d = json.loads(frame_data)
            b = base64.b64decode(d["frame"])
            arr = np.frombuffer(b, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            payload = {"timestamp": d.get("timestamp", time.time()), "frame": frame}

            if self._queue.full():
                try:
                    self._queue.get_nowait()  # drop oldest
                except Exception:
                    pass
            self._queue.put_nowait(payload)

        except Exception as e:
            logger.warning(f"[Detect] enqueue error: {e}")

    def _process_item_sync(self, item: dict[str, Any]) -> None:
        """
        Synchronous heavy processing for a single queued frame:
        - YOLO detection
        - publish latest {ts, frame, detections} into IOProvider
        """
        frame = item["frame"]
        ts = now_ms()

        dets = self.detector.infer(frame)
        logging.debug(
            f"[Detect] YOLO returned {len(dets)} detections for frame ts={ts}"
        )

        packet = {
            "ts": ts,
            "frame": frame,
            "detections": dets,
        }

        # Publish into IOProvider
        try:
            self.io_provider.add_dynamic_variable(self.IO_KEY_LATEST, packet)
        except Exception:
            logger.exception("[Detect] Failed to publish detections into IOProvider")

    async def _consumer(self):
        """
        Background worker that pulls frames and offloads YOLO to a separate thread.
        """
        try:
            while True:
                item = await self._queue.get()
                await asyncio.to_thread(self._process_item_sync, item)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"[Detect] worker error: {e}")
