"""Tests for the VLM Privacy Detector input plugin."""

import time
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.vlm_privacy_detector import (
    DEFAULT_PRIVACY_CLASSES,
    SENSITIVITY_THRESHOLDS,
    THREAT_SEVERITY,
    VLMPrivacyDetector,
    VLMPrivacyDetectorConfig,
)


@pytest.fixture
def config():
    return VLMPrivacyDetectorConfig(
        camera_index=0,
        log_file=False,
        privacy_classes=list(DEFAULT_PRIVACY_CLASSES),
        sensitivity="high",
    )


@pytest.fixture
def detector(config):
    """Create a detector with mocked camera and YOLO model."""
    with patch("inputs.plugins.vlm_privacy_detector._init_camera", return_value=(640, 480)), \
         patch("inputs.plugins.vlm_privacy_detector.cv2") as mock_cv2, \
         patch("inputs.plugins.vlm_privacy_detector.YOLO") as mock_yolo, \
         patch("inputs.plugins.vlm_privacy_detector.IOProvider"):
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        det = VLMPrivacyDetector(config)
        det.have_cam = True
        det.cam_third = 213  # 640 / 3
        yield det


class TestThreatSeverity:
    """Test the severity classification constants."""

    def test_high_severity_classes(self):
        assert THREAT_SEVERITY["laptop"] == "HIGH"
        assert THREAT_SEVERITY["tv"] == "HIGH"
        assert THREAT_SEVERITY["keyboard"] == "HIGH"

    def test_medium_severity_classes(self):
        assert THREAT_SEVERITY["cell phone"] == "MEDIUM"

    def test_low_severity_classes(self):
        assert THREAT_SEVERITY["mouse"] == "LOW"
        assert THREAT_SEVERITY["book"] == "LOW"
        assert THREAT_SEVERITY["remote"] == "LOW"


class TestSensitivityThresholds:
    """Test sensitivity threshold configuration."""

    def test_thresholds(self):
        assert SENSITIVITY_THRESHOLDS["high"] == 0.3
        assert SENSITIVITY_THRESHOLDS["medium"] == 0.5
        assert SENSITIVITY_THRESHOLDS["low"] == 0.7


class TestDirectionLabel:
    """Test the directional labeling logic."""

    def test_left(self, detector):
        assert detector._direction_label(0, 100) == "on your left"

    def test_center(self, detector):
        assert detector._direction_label(250, 400) == "in front of you"

    def test_right(self, detector):
        assert detector._direction_label(500, 640) == "on your right"


class TestClassifyThreat:
    """Test the threat classification pipeline."""

    def test_no_detections(self, detector):
        result = detector._classify_threat([])
        assert result["threats"] == []
        assert result["safe"] == []
        assert result["max_severity"] == "NONE"

    def test_high_threat(self, detector):
        detections = [
            {"class": "laptop", "confidence": 0.85, "bbox": [100, 100, 300, 300]},
        ]
        result = detector._classify_threat(detections)
        assert len(result["threats"]) == 1
        assert result["max_severity"] == "HIGH"
        assert result["threats"][0]["severity"] == "HIGH"

    def test_low_threat(self, detector):
        detections = [
            {"class": "mouse", "confidence": 0.9, "bbox": [100, 100, 200, 200]},
        ]
        result = detector._classify_threat(detections)
        assert len(result["threats"]) == 1
        assert result["max_severity"] == "LOW"

    def test_mixed_detections(self, detector):
        detections = [
            {"class": "laptop", "confidence": 0.85, "bbox": [100, 100, 300, 300]},
            {"class": "cell phone", "confidence": 0.7, "bbox": [350, 100, 450, 300]},
            {"class": "person", "confidence": 0.95, "bbox": [0, 0, 640, 480]},
        ]
        result = detector._classify_threat(detections)
        assert len(result["threats"]) == 2
        assert len(result["safe"]) == 1
        assert result["max_severity"] == "HIGH"

    def test_below_threshold_ignored(self, detector):
        """Detections below the confidence threshold should be classified as safe."""
        detections = [
            {"class": "laptop", "confidence": 0.1, "bbox": [100, 100, 300, 300]},
        ]
        result = detector._classify_threat(detections)
        assert len(result["threats"]) == 0
        assert len(result["safe"]) == 1
        assert result["max_severity"] == "NONE"

    def test_non_privacy_class_is_safe(self, detector):
        """Classes not in the privacy list should always be safe."""
        detections = [
            {"class": "person", "confidence": 0.99, "bbox": [100, 100, 300, 300]},
        ]
        result = detector._classify_threat(detections)
        assert len(result["threats"]) == 0
        assert len(result["safe"]) == 1


class TestRawToText:
    """Test the raw-to-text conversion."""

    @pytest.mark.asyncio
    async def test_no_input_returns_none(self, detector):
        result = await detector._raw_to_text(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_input_after_threat_returns_clear(self, detector):
        detector._threat_streak = 3
        result = await detector._raw_to_text(None)
        assert result is not None
        assert "CLEAR" in result.message

    @pytest.mark.asyncio
    async def test_threat_generates_alert(self, detector):
        detections = [
            {"class": "laptop", "confidence": 0.85, "bbox": [100, 100, 300, 300]},
        ]
        result = await detector._raw_to_text(detections)
        assert result is not None
        assert "PRIVACY ALERT" in result.message
        assert "HIGH" in result.message
        assert detector._threat_streak == 1

    @pytest.mark.asyncio
    async def test_safe_detections_return_clear(self, detector):
        detections = [
            {"class": "person", "confidence": 0.9, "bbox": [100, 100, 300, 300]},
        ]
        result = await detector._raw_to_text(detections)
        assert result is not None
        assert "CLEAR" in result.message

    @pytest.mark.asyncio
    async def test_streak_increments(self, detector):
        detections = [
            {"class": "laptop", "confidence": 0.85, "bbox": [100, 100, 300, 300]},
        ]
        await detector._raw_to_text(detections)
        assert detector._threat_streak == 1
        await detector._raw_to_text(detections)
        assert detector._threat_streak == 2

    @pytest.mark.asyncio
    async def test_streak_resets_on_clear(self, detector):
        detections = [
            {"class": "laptop", "confidence": 0.85, "bbox": [100, 100, 300, 300]},
        ]
        await detector._raw_to_text(detections)
        assert detector._threat_streak == 1

        safe = [
            {"class": "person", "confidence": 0.9, "bbox": [100, 100, 300, 300]},
        ]
        await detector._raw_to_text(safe)
        assert detector._threat_streak == 0
