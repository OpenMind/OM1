import time
from unittest.mock import MagicMock, patch

import pytest

from providers.ubtech_video_stream import UbtechCameraVideoStream


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for UbtechCameraVideoStream."""
    with (
        patch("providers.ubtech_video_stream.YanAPI") as mock_yan_api,
        patch("providers.ubtech_video_stream.MJPEGClient") as mock_mjpeg_client,
        patch("providers.ubtech_video_stream.cv2") as mock_cv2,
    ):

        mock_client_instance = MagicMock()
        mock_mjpeg_client.return_value = mock_client_instance

        yield {
            "yan_api": mock_yan_api,
            "mjpeg_client": mock_mjpeg_client,
            "client_instance": mock_client_instance,
            "cv2": mock_cv2,
        }


def test_initialization(mock_dependencies):
    """Test UbtechCameraVideoStream initialization."""
    robot_ip = "192.168.1.100"

    stream = UbtechCameraVideoStream(
        robot_ip=robot_ip, fps=30, resolution=(640, 480), jpeg_quality=70
    )

    assert stream.robot_ip == robot_ip
    assert stream.url == f"http://{robot_ip}:8000/stream.mjpg"
    assert stream.stream_client is None
    mock_dependencies["yan_api"].yan_api_init.assert_called_once_with(robot_ip)


def test_initialization_with_callback(mock_dependencies):
    """Test initialization with a frame callback."""

    def callback(frame):
        pass

    stream = UbtechCameraVideoStream(robot_ip="192.168.1.100", frame_callback=callback)

    assert callback in stream.frame_callbacks


def test_initialization_with_multiple_callbacks(mock_dependencies):
    """Test initialization with multiple frame callbacks."""

    def callback1(frame):
        pass

    def callback2(frame):
        pass

    callbacks = [callback1, callback2]

    stream = UbtechCameraVideoStream(
        robot_ip="192.168.1.100", frame_callbacks=callbacks
    )

    assert callback1 in stream.frame_callbacks
    assert callback2 in stream.frame_callbacks


def test_default_fps(mock_dependencies):
    """Test default FPS value."""
    stream = UbtechCameraVideoStream(robot_ip="192.168.1.100")

    assert stream.fps == 30


def test_custom_fps(mock_dependencies):
    """Test custom FPS value."""
    stream = UbtechCameraVideoStream(robot_ip="192.168.1.100", fps=15)

    assert stream.fps == 15


def test_default_resolution(mock_dependencies):
    """Test default resolution value."""
    stream = UbtechCameraVideoStream(robot_ip="192.168.1.100")

    assert stream.resolution == (640, 480)


def test_custom_resolution(mock_dependencies):
    """Test custom resolution value."""
    stream = UbtechCameraVideoStream(robot_ip="192.168.1.100", resolution=(1280, 720))

    assert stream.resolution == (1280, 720)


def test_jpeg_quality(mock_dependencies):
    """Test JPEG quality setting."""
    stream = UbtechCameraVideoStream(robot_ip="192.168.1.100", jpeg_quality=85)

    assert stream.encode_quality == [1, 85]


def test_url_format(mock_dependencies):
    """Test URL format generation."""
    robot_ip = "10.0.0.50"
    stream = UbtechCameraVideoStream(robot_ip=robot_ip)

    expected_url = f"http://{robot_ip}:8000/stream.mjpg"
    assert stream.url == expected_url


def test_yan_api_initialization(mock_dependencies):
    """Test that YanAPI is initialized on creation."""
    robot_ip = "192.168.1.100"
    UbtechCameraVideoStream(robot_ip=robot_ip)

    mock_dependencies["yan_api"].yan_api_init.assert_called_once_with(robot_ip)


def test_start_calls_on_video(mock_dependencies):
    """Test that start method initiates the video stream."""
    stream = UbtechCameraVideoStream(robot_ip="192.168.1.100")

    with patch.object(stream, "on_video"):
        stream.start()

        time.sleep(0.1)
        stream.stop()


def test_multiple_frame_callbacks(mock_dependencies):
    """Test adding multiple frame callbacks."""
    callback1 = MagicMock()
    callback2 = MagicMock()

    stream = UbtechCameraVideoStream(
        robot_ip="192.168.1.100", frame_callbacks=[callback1, callback2]
    )

    assert len(stream.frame_callbacks) >= 2
    assert callback1 in stream.frame_callbacks
    assert callback2 in stream.frame_callbacks
