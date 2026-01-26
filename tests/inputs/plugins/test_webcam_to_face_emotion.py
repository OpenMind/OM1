from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.plugins.webcam_to_face_emotion import FaceEmotionCapture


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.webcam_to_face_emotion.IOProvider"),
        patch("inputs.plugins.webcam_to_face_emotion.cv2.CascadeClassifier"),
        patch("inputs.plugins.webcam_to_face_emotion.check_webcam", return_value=True),
        patch("inputs.plugins.webcam_to_face_emotion.cv2.VideoCapture"),
    ):
        config = SensorConfig()
        sensor = FaceEmotionCapture(config=config)

        assert hasattr(sensor, "messages")


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, "mock_frame")  # (ret, frame)

    with (
        patch("inputs.plugins.webcam_to_face_emotion.IOProvider"),
        patch("inputs.plugins.webcam_to_face_emotion.cv2.CascadeClassifier"),
        patch("inputs.plugins.webcam_to_face_emotion.check_webcam", return_value=True),
        patch(
            "inputs.plugins.webcam_to_face_emotion.cv2.VideoCapture",
            return_value=mock_cap,
        ),
        patch("inputs.plugins.webcam_to_face_emotion.asyncio.sleep", new=AsyncMock()),
    ):
        config = SensorConfig()
        sensor = FaceEmotionCapture(config=config)

        result = await sensor._poll()
        assert result == "mock_frame"


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.webcam_to_face_emotion.IOProvider"),
        patch("inputs.plugins.webcam_to_face_emotion.cv2.CascadeClassifier"),
        patch("inputs.plugins.webcam_to_face_emotion.check_webcam", return_value=True),
        patch("inputs.plugins.webcam_to_face_emotion.cv2.VideoCapture"),
    ):
        config = SensorConfig()
        sensor = FaceEmotionCapture(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(
            timestamp=123.456, message="I see a person. Their emotion is happy."
        )
        sensor.messages.append(test_message)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "FaceEmotionCapture INPUT" in result
        assert "I see a person. Their emotion is happy." in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0
