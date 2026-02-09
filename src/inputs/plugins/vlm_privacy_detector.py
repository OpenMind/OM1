"""VLM Privacy Detector input plugin.

Extends YOLO object detection with privacy-specific classification.
Instead of reporting "You see a laptop on your left", this plugin
generates structured privacy threat assessments like:

    "PRIVACY ALERT [HIGH]: laptop screen detected on your left.
     Additional objects: cell phone (center), keyboard (right).
     Threat level: HIGH — Immediate action required."

The LLM cortex receives these enriched captions and uses the Privacy
Guard system prompt to decide the appropriate action (look_away, etc.).
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

import cv2
from pydantic import Field
from ultralytics import YOLO  # type: ignore

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

logger = logging.getLogger(__name__)

# YOLO COCO class names that indicate privacy-sensitive objects
DEFAULT_PRIVACY_CLASSES = {
    "laptop",
    "tv",
    "cell phone",
    "keyboard",
    "mouse",
    "book",
    "remote",
}

# Severity mapping: class name → threat level
THREAT_SEVERITY: Dict[str, str] = {
    "laptop": "HIGH",
    "tv": "HIGH",
    "cell phone": "MEDIUM",
    "keyboard": "HIGH",
    "mouse": "LOW",
    "book": "LOW",
    "remote": "LOW",
}


class VLMPrivacyDetectorConfig(SensorConfig):
    """Configuration for the Privacy Detector VLM sensor.

    Parameters
    ----------
    camera_index : int
        Index of the camera device.
    log_file : bool
        Whether to enable JSONL file logging for forensics.
    privacy_classes : list
        YOLO class names to flag as privacy-sensitive.
    sensitivity : str
        Detection sensitivity: "low", "medium", or "high".
        Controls the confidence threshold for flagging objects.
    """

    camera_index: int = Field(default=0, description="Camera device index")
    log_file: bool = Field(default=False, description="Enable JSONL logging")
    privacy_classes: list = Field(
        default_factory=lambda: list(DEFAULT_PRIVACY_CLASSES),
        description="YOLO class names to flag as privacy threats",
    )
    sensitivity: str = Field(
        default="high",
        description="Detection sensitivity: low (>0.7), medium (>0.5), high (>0.3)",
    )


# Common resolutions to test, ordered high to low
RESOLUTIONS = [
    (1920, 1080),
    (1280, 720),
    (640, 480),
]

SENSITIVITY_THRESHOLDS = {
    "low": 0.7,
    "medium": 0.5,
    "high": 0.3,
}


def _init_camera(index: int) -> tuple:
    """Check webcam availability and find best resolution."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error(f"Privacy Detector: camera {index} not found")
        return 0, 0
    for w, h in RESOLUTIONS:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        time.sleep(0.1)
        if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == w:
            logger.info(f"Privacy Detector: camera {index} set to {w}x{h}")
            cap.release()
            return w, h
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


class VLMPrivacyDetector(FuserInput[VLMPrivacyDetectorConfig, Optional[List]]):
    """VLM input handler with privacy-threat enrichment on top of YOLO."""

    def __init__(self, config: VLMPrivacyDetectorConfig):
        super().__init__(config)

        self.camera_index = config.camera_index
        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "PrivacyEyes"

        # Load YOLO model
        self.model = YOLO("yolov8n.pt")

        # Privacy config
        self.privacy_classes = set(config.privacy_classes)
        self.confidence_threshold = SENSITIVITY_THRESHOLDS.get(
            config.sensitivity, 0.3
        )

        # Logging
        self.write_to_local_file = config.log_file
        self.filename_current = None
        self.max_file_size_bytes = 1024 * 1024
        if self.write_to_local_file:
            ts = str(round(time.time(), 6)).replace(".", "_")
            self.filename_current = f"dump/privacy_{ts}Z.jsonl"
            logger.info(f"Privacy logging to {self.filename_current}")

        # Camera
        self.width, self.height = _init_camera(self.camera_index)
        self.have_cam = self.width > 0
        self.cap = None
        if self.have_cam:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cam_third = int(self.width / 3)

        self.frame_index = 0

        # State tracking — how many consecutive frames had a threat
        self._threat_streak = 0
        self._last_threat_level = "NONE"

    def _direction_label(self, x1: float, x2: float) -> str:
        """Convert bbox horizontal center to a directional label."""
        center = (x1 + x2) / 2
        if center < self.cam_third:
            return "on your left"
        elif center > 2 * self.cam_third:
            return "on your right"
        return "in front of you"

    def _classify_threat(self, detections: List[dict]) -> dict:
        """Partition detections into privacy threats vs. safe objects.

        Returns
        -------
        dict with keys: threats (list), safe (list), max_severity (str)
        """
        threats = []
        safe = []
        max_severity = "NONE"
        severity_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

        for det in detections:
            cls_name = det["class"]
            if cls_name in self.privacy_classes and det["confidence"] >= self.confidence_threshold:
                severity = THREAT_SEVERITY.get(cls_name, "MEDIUM")
                det["severity"] = severity
                det["direction"] = self._direction_label(det["bbox"][0], det["bbox"][2])
                threats.append(det)
                if severity_rank.get(severity, 0) > severity_rank.get(max_severity, 0):
                    max_severity = severity
            else:
                safe.append(det)

        return {"threats": threats, "safe": safe, "max_severity": max_severity}

    async def _poll(self) -> Optional[List]:
        """Capture a frame and run YOLO inference."""
        await asyncio.sleep(0.25)

        if not self.have_cam or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_index += 1
        results = self.model.predict(source=frame, save=False, stream=True, verbose=False)

        detections = []
        for r in results:
            if r.boxes is not None:
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

        # Write forensic log
        if self.write_to_local_file and self.filename_current:
            try:
                line = json.dumps(
                    {
                        "frame": self.frame_index,
                        "timestamp": time.time(),
                        "detections": detections,
                    }
                )
                if (
                    os.path.exists(self.filename_current)
                    and os.path.getsize(self.filename_current) > self.max_file_size_bytes
                ):
                    ts = str(round(time.time(), 6)).replace(".", "_")
                    self.filename_current = f"dump/privacy_{ts}Z.jsonl"
                with open(self.filename_current, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.error(f"Privacy log write error: {e}")

        return detections

    async def _raw_to_text(self, raw_input: Optional[List]) -> Optional[Message]:
        """Convert YOLO detections into a privacy-enriched text report."""
        if not raw_input:
            # No detections at all — report clear environment
            if self._threat_streak > 0:
                self._threat_streak = 0
                self._last_threat_level = "NONE"
                return Message(
                    timestamp=time.time(),
                    message="PRIVACY STATUS: Environment is now CLEAR. No objects detected.",
                )
            return None

        analysis = self._classify_threat(raw_input)
        threats = analysis["threats"]
        safe = analysis["safe"]
        max_severity = analysis["max_severity"]

        if not threats:
            # Objects detected but none are privacy-sensitive
            self._threat_streak = 0
            self._last_threat_level = "NONE"
            top = max(raw_input, key=lambda d: d["confidence"])
            direction = self._direction_label(top["bbox"][0], top["bbox"][2])
            return Message(
                timestamp=time.time(),
                message=f"PRIVACY STATUS: CLEAR. You see a {top['class']} {direction}. No privacy threats detected.",
            )

        # Privacy threats found
        self._threat_streak += 1
        self._last_threat_level = max_severity

        # Build threat report
        threat_lines = []
        for t in sorted(threats, key=lambda x: -SENSITIVITY_THRESHOLDS.get(x.get("severity", "MEDIUM"), 0.5)):
            threat_lines.append(
                f"  - {t['class']} [{t['severity']}] ({t['confidence']:.0%} confidence) {t['direction']}"
            )

        report = (
            f"PRIVACY ALERT [{max_severity}]: {len(threats)} privacy-sensitive object(s) detected!\n"
            + "\n".join(threat_lines)
            + f"\nThreat streak: {self._threat_streak} consecutive frame(s)."
            + f"\nAction required: {'IMMEDIATE' if max_severity == 'HIGH' else 'EVALUATE'}."
        )

        if safe:
            safe_names = [s["class"] for s in safe[:3]]
            report += f"\nOther objects in view: {', '.join(safe_names)}."

        logger.warning(f"Privacy threat detected: {max_severity} ({len(threats)} objects)")
        return Message(timestamp=time.time(), message=report)

    async def raw_to_text(self, raw_input: Optional[List]):
        """Convert detections to text and buffer."""
        pending = await self._raw_to_text(raw_input)
        if pending is not None:
            self.messages.append(pending)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format the latest buffered message for the LLM cortex."""
        if not self.messages:
            return None

        latest = self.messages[-1]
        logger.info(f"PrivacyDetector: {latest.message}")

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest.message, latest.timestamp
        )
        self.messages = []
        return result
