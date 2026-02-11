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
