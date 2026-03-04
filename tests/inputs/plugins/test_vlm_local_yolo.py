from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.vlm_local_yolo import VLM_Local_YOLO, VLM_Local_YOLOConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture"),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, MagicMock())  # (ret, frame)

    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture", return_value=mock_cap),
        patch("inputs.plugins.vlm_local_yolo.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        result = await sensor._poll()
        assert result == []


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture"),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456, message="You see a person in front of you."
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Eyes" in result
        assert "You see a person" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0


@pytest.mark.asyncio
async def test_raw_to_text_with_detections():
    """Test raw_to_text with YOLO detections adds message to buffer."""
    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture"),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        detections = [
            {"class": "person", "confidence": 0.95, "bbox": [100, 50, 300, 400]},
            {"class": "cat", "confidence": 0.80, "bbox": [400, 100, 500, 300]},
        ]
        await sensor.raw_to_text(detections)
        assert len(sensor.messages) == 1
        assert "person" in sensor.messages[0].message


@pytest.mark.asyncio
async def test_raw_to_text_empty_detections():
    """Test raw_to_text with empty detections does not add message."""
    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture"),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        await sensor.raw_to_text([])
        assert len(sensor.messages) == 0


def test_get_top_detection_with_detections():
    """Test get_top_detection returns highest confidence detection."""
    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture"),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        detections = [
            {"class": "cat", "confidence": 0.80, "bbox": [10, 20, 30, 40]},
            {"class": "person", "confidence": 0.95, "bbox": [100, 200, 300, 400]},
            {"class": "dog", "confidence": 0.70, "bbox": [50, 60, 70, 80]},
        ]
        label, bbox = sensor.get_top_detection(detections)
        assert label == "person"
        assert bbox == [100, 200, 300, 400]


def test_get_top_detection_empty():
    """Test get_top_detection with empty list returns (None, None)."""
    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture"),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        label, bbox = sensor.get_top_detection([])
        assert label is None
        assert bbox is None


@pytest.mark.asyncio
async def test_poll_returns_none_on_failed_frame_read():
    """Test that _poll returns None when cap.read() fails."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture", return_value=mock_cap),
        patch("inputs.plugins.vlm_local_yolo.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_poll_returns_none_on_none_frame():
    """Test that _poll returns None when cap.read() returns True but frame is None."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, None)

    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(640, 480)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture", return_value=mock_cap),
        patch("inputs.plugins.vlm_local_yolo.asyncio.sleep", new=AsyncMock()),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        result = await sensor._poll()
        assert result is None


def test_cam_third_initialized_without_camera():
    """Test that cam_third has a safe default when no camera is available."""
    with (
        patch("inputs.plugins.vlm_local_yolo.IOProvider"),
        patch("inputs.plugins.vlm_local_yolo.YOLO"),
        patch("inputs.plugins.vlm_local_yolo.check_webcam", return_value=(0, 0)),
        patch("inputs.plugins.vlm_local_yolo.cv2.VideoCapture"),
    ):
        config = VLM_Local_YOLOConfig()
        sensor = VLM_Local_YOLO(config=config)

        assert hasattr(sensor, "cam_third")
        assert sensor.cam_third == 0
