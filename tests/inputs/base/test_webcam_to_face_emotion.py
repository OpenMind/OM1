from unittest.mock import Mock, patch

import numpy as np
import pytest

from inputs.base import SensorConfig
from inputs.plugins.webcam_to_face_emotion import FaceEmotionCapture, Message


@pytest.fixture
def mock_cv2():
    with patch("inputs.plugins.webcam_to_face_emotion.cv2") as mock:
        mock.data.haarcascades = "/mock/path/"
        mock.CascadeClassifier.return_value = Mock()
        mock_capture = Mock()
        mock_capture.read.return_value = (True, np.zeros((100, 100, 3)))
        mock.VideoCapture.return_value = mock_capture
        mock.cvtColor = Mock(return_value=np.zeros((100, 100, 3)))
        yield mock


@pytest.fixture
def mock_deepface():
    with patch("inputs.plugins.webcam_to_face_emotion.DeepFace") as mock:
        mock.analyze.return_value = [{"dominant_emotion": "happy"}]
        yield mock


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.webcam_to_face_emotion.IOProvider") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def face_emotion(mock_cv2, mock_io_provider, mock_deepface):
    with patch("inputs.plugins.webcam_to_face_emotion.check_webcam", return_value=True):
        instance = FaceEmotionCapture(config=SensorConfig())
        instance.face_cascade.detectMultiScale = Mock(return_value=[(10, 10, 50, 50)])
        instance.have_cam = True
        mock_cap = Mock()
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3)))
        instance.cap = mock_cap
        return instance


def test_init(face_emotion, mock_cv2):
    assert isinstance(face_emotion.messages, list)
    assert face_emotion.emotion == ""
    mock_cv2.CascadeClassifier.assert_called_once()
    mock_cv2.VideoCapture.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_poll(face_emotion):
    face_emotion.cap.read.return_value = (True, np.zeros((100, 100, 3)))
    result = await face_emotion._poll()
    assert isinstance(result, np.ndarray)
    face_emotion.cap.read.assert_called_once()


@pytest.mark.asyncio
async def test_raw_to_text_with_face(face_emotion, mock_cv2, mock_deepface):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    face_emotion.face_cascade.detectMultiScale.return_value = [(10, 10, 50, 50)]

    result = await face_emotion._raw_to_text(frame)

    assert isinstance(result, Message)
    assert "happy" in result.message
    mock_deepface.analyze.assert_called_once()


@pytest.mark.asyncio
async def test_raw_to_text_no_face(face_emotion, mock_cv2, mock_deepface):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    face_emotion.face_cascade.detectMultiScale.return_value = []

    result = await face_emotion._raw_to_text(frame)

    assert isinstance(result, Message)
    mock_deepface.analyze.assert_not_called()


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management(face_emotion, mock_deepface):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deepface.analyze.return_value = [{"dominant_emotion": "happy"}]

    await face_emotion.raw_to_text(frame)
    assert len(face_emotion.messages) == 1
    assert isinstance(face_emotion.messages[0], Message)
    assert "happy" in face_emotion.messages[0].message


def test_formatted_latest_buffer_with_message(face_emotion):
    test_message = Message(timestamp=123.456, message="test emotion")
    face_emotion.messages = [test_message]

    result = face_emotion.formatted_latest_buffer()

    assert "FaceEmotionCapture INPUT" in result
    assert len(face_emotion.messages) == 0
    face_emotion.io_provider.add_input.assert_called_once_with(
        "FaceEmotionCapture", "test emotion", 123.456
    )


def test_formatted_latest_buffer_empty(face_emotion):
    assert face_emotion.formatted_latest_buffer() is None
    face_emotion.io_provider.add_input.assert_not_called()
