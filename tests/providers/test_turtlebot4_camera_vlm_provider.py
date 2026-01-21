from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from providers.turtlebot4_camera_vlm_provider import (
    TurtleBot4CameraVideoStream,
    TurtleBot4CameraVLMProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TurtleBot4CameraVLMProvider.reset()  # type: ignore
    yield

    try:
        provider = getattr(TurtleBot4CameraVLMProvider, "_instance", None)
        if provider:
            provider.stop()
    except Exception:
        pass

    TurtleBot4CameraVLMProvider.reset()  # type: ignore


@pytest.fixture
def mock_zenoh():
    """Mock Zenoh dependencies."""
    with patch(
        "providers.turtlebot4_camera_vlm_provider.open_zenoh_session"
    ) as mock_session:
        mock_session_instance = MagicMock()
        mock_subscriber = MagicMock()
        mock_session_instance.declare_subscriber.return_value = mock_subscriber
        mock_session.return_value = mock_session_instance
        yield mock_session, mock_session_instance, mock_subscriber


def test_turtlebot4_video_stream_initialization(mock_zenoh):
    """Test TurtleBot4CameraVideoStream initialization."""
    _, mock_session_instance, _ = mock_zenoh

    stream = TurtleBot4CameraVideoStream(
        fps=30, resolution=(640, 480), jpeg_quality=70, URID="test_robot"
    )

    assert stream.session == mock_session_instance
    assert stream.debug is False
    assert stream.image is None


def test_turtlebot4_video_stream_with_debug(mock_zenoh):
    """Test TurtleBot4CameraVideoStream with debug mode."""
    stream = TurtleBot4CameraVideoStream(URID="test_robot", debug=True)

    assert stream.debug is True


def test_turtlebot4_video_stream_default_urid(mock_zenoh):
    """Test default URID value."""
    TurtleBot4CameraVideoStream()

    mock_zenoh[1].declare_subscriber.assert_called_once()
    call_args = mock_zenoh[1].declare_subscriber.call_args
    assert "default/pi/oakd/rgb/preview/image_raw" in call_args[0][0]


def test_turtlebot4_video_stream_custom_urid(mock_zenoh):
    """Test custom URID value."""
    TurtleBot4CameraVideoStream(URID="robot123")

    mock_zenoh[1].declare_subscriber.assert_called_once()
    call_args = mock_zenoh[1].declare_subscriber.call_args
    assert "robot123/pi/oakd/rgb/preview/image_raw" in call_args[0][0]


def test_turtlebot4_video_stream_with_callback(mock_zenoh):
    """Test initialization with frame callback."""

    def callback(frame):
        pass

    stream = TurtleBot4CameraVideoStream(frame_callback=callback)

    assert callback in stream.frame_callbacks


def test_turtlebot4_video_stream_initialization_failure():
    """Test handling of Zenoh initialization failure."""
    with patch(
        "providers.turtlebot4_camera_vlm_provider.open_zenoh_session"
    ) as mock_session:
        mock_session.side_effect = Exception("Connection failed")

        stream = TurtleBot4CameraVideoStream()

        assert stream.session is None


def test_turtlebot4_video_stream_camera_listener(mock_zenoh):
    """Test camera listener with valid data."""
    stream = TurtleBot4CameraVideoStream()

    mock_sample = MagicMock()
    mock_payload = MagicMock()
    data = np.zeros(187576, dtype=np.uint8)
    mock_payload.to_bytes.return_value = data.tobytes()
    mock_sample.payload = mock_payload

    with patch("providers.turtlebot4_camera_vlm_provider.cv2"):
        stream.camera_listener(mock_sample)

        assert stream.image is not None


def test_camera_listener_with_debug(mock_zenoh):
    """Test camera listener in debug mode."""
    stream = TurtleBot4CameraVideoStream(debug=True)

    mock_sample = MagicMock()
    mock_payload = MagicMock()

    data = np.zeros(187576, dtype=np.uint8)
    mock_payload.to_bytes.return_value = data.tobytes()
    mock_sample.payload = mock_payload

    with patch("providers.turtlebot4_camera_vlm_provider.cv2") as mock_cv2:
        stream.camera_listener(mock_sample)

        mock_cv2.imwrite.assert_called_once()


def test_turtlebot4_vlm_provider_initialization():
    """Test TurtleBot4CameraVLMProvider initialization."""
    with (
        patch("providers.turtlebot4_camera_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.turtlebot4_camera_vlm_provider.TurtleBot4CameraVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        provider = TurtleBot4CameraVLMProvider(
            ws_url="ws://localhost:8000", URID="test_robot", fps=15
        )

        assert provider.running is False
        mock_ws.assert_called_once_with(url="ws://localhost:8000")


def test_turtlebot4_vlm_provider_singleton():
    """Test that TurtleBot4CameraVLMProvider follows singleton pattern."""
    with (
        patch("providers.turtlebot4_camera_vlm_provider.ws.Client"),
        patch("providers.turtlebot4_camera_vlm_provider.TurtleBot4CameraVideoStream"),
    ):

        provider1 = TurtleBot4CameraVLMProvider(
            ws_url="ws://localhost:8000", URID="robot1"
        )
        provider2 = TurtleBot4CameraVLMProvider(
            ws_url="ws://localhost:8001", URID="robot2"
        )
        assert provider1 is provider2


def test_turtlebot4_vlm_provider_start():
    """Test starting the VLM provider."""
    with (
        patch("providers.turtlebot4_camera_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.turtlebot4_camera_vlm_provider.TurtleBot4CameraVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        provider = TurtleBot4CameraVLMProvider(
            ws_url="ws://localhost:8000", URID="test"
        )
        provider.start()

        assert provider.running is True
        mock_ws_instance.start.assert_called_once()
        mock_stream_instance.start.assert_called_once()


def test_turtlebot4_vlm_provider_stop():
    """Test stopping the VLM provider."""
    with (
        patch("providers.turtlebot4_camera_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.turtlebot4_camera_vlm_provider.TurtleBot4CameraVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        provider = TurtleBot4CameraVLMProvider(
            ws_url="ws://localhost:8000", URID="test"
        )
        provider.start()
        provider.stop()

        assert provider.running is False
        mock_stream_instance.stop.assert_called_once()
        mock_ws_instance.stop.assert_called_once()
