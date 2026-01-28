from unittest.mock import MagicMock, patch

import pytest

from providers.unitree_realsense_dev_vlm_provider import (
    UnitreeRealSenseDevVideoStream,
    UnitreeRealSenseDevVLMProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeRealSenseDevVLMProvider.reset()  # type: ignore
    yield

    try:
        provider = getattr(UnitreeRealSenseDevVLMProvider, "_instance", None)
        if provider:
            provider.stop()
    except Exception:
        pass

    UnitreeRealSenseDevVLMProvider.reset()  # type: ignore


@pytest.fixture
def mock_cv2():
    """Mock OpenCV."""
    with patch("providers.unitree_realsense_dev_vlm_provider.cv2") as mock:
        yield mock


def test_video_stream_initialization(mock_cv2):
    """Test UnitreeRealSenseDevVideoStream initialization."""
    stream = UnitreeRealSenseDevVideoStream(
        fps=30, resolution=(640, 480), jpeg_quality=70
    )

    assert stream.fps == 30
    assert stream.resolution == (640, 480)
    assert stream.encode_quality == [1, 70]


def test_video_stream_with_callback(mock_cv2):
    """Test initialization with callback."""

    def callback(frame):
        pass

    stream = UnitreeRealSenseDevVideoStream(frame_callback=callback)

    assert callback in stream.frame_callbacks


def test_video_stream_with_multiple_callbacks(mock_cv2):
    """Test initialization with multiple callbacks."""

    def callback1(frame):
        pass

    def callback2(frame):
        pass

    stream = UnitreeRealSenseDevVideoStream(frame_callbacks=[callback1, callback2])

    assert callback1 in stream.frame_callbacks
    assert callback2 in stream.frame_callbacks


def test_video_stream_default_fps(mock_cv2):
    """Test default FPS value."""
    stream = UnitreeRealSenseDevVideoStream()

    assert stream.fps == 30


def test_video_stream_custom_fps(mock_cv2):
    """Test custom FPS value."""
    stream = UnitreeRealSenseDevVideoStream(fps=15)

    assert stream.fps == 15


def test_video_stream_default_resolution(mock_cv2):
    """Test default resolution."""
    stream = UnitreeRealSenseDevVideoStream()

    assert stream.resolution == (640, 480)


def test_video_stream_custom_resolution(mock_cv2):
    """Test custom resolution."""
    stream = UnitreeRealSenseDevVideoStream(resolution=(1280, 720))

    assert stream.resolution == (1280, 720)


def test_video_stream_jpeg_quality(mock_cv2):
    """Test JPEG quality setting."""
    stream = UnitreeRealSenseDevVideoStream(jpeg_quality=85)

    assert stream.encode_quality == [1, 85]


def test_vlm_provider_initialization():
    """Test UnitreeRealSenseDevVLMProvider initialization."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        provider = UnitreeRealSenseDevVLMProvider(
            ws_url="ws://localhost:8000", fps=30, resolution=(640, 480), jpeg_quality=70
        )

        assert provider.running is False
        mock_ws.assert_called_once_with(url="ws://localhost:8000")


def test_vlm_provider_singleton():
    """Test that UnitreeRealSenseDevVLMProvider follows singleton pattern."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client"),
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ),
    ):

        provider1 = UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:8000")
        provider2 = UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:9000")
        assert provider1 is provider2


def test_vlm_provider_register_message_callback():
    """Test registering message callback."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ),
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        provider = UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:8000")

        def callback(message):
            pass

        provider.register_message_callback(callback)

        mock_ws_instance.register_message_callback.assert_called_once_with(callback)


def test_vlm_provider_register_message_callback_none():
    """Test registering None callback."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ),
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        provider = UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:8000")

        provider.register_message_callback(None)

        mock_ws_instance.register_message_callback.assert_not_called()


def test_vlm_provider_start():
    """Test starting the VLM provider."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        provider = UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:8000")
        provider.start()

        assert provider.running is True
        mock_ws_instance.start.assert_called_once()
        mock_stream_instance.start.assert_called_once()


def test_vlm_provider_start_already_running():
    """Test starting when already running."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        provider = UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:8000")
        provider.start()

        mock_ws_instance.start.reset_mock()
        mock_stream_instance.start.reset_mock()

        provider.start()

        mock_ws_instance.start.assert_not_called()
        mock_stream_instance.start.assert_not_called()


def test_vlm_provider_stop():
    """Test stopping the VLM provider."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        provider = UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:8000")
        provider.start()
        provider.stop()

        assert provider.running is False
        mock_stream_instance.stop.assert_called_once()
        mock_ws_instance.stop.assert_called_once()


def test_vlm_provider_video_stream_parameters():
    """Test that video stream is created with correct parameters."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client"),
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ) as mock_stream,
    ):

        UnitreeRealSenseDevVLMProvider(
            ws_url="ws://localhost:8000",
            fps=15,
            resolution=(1280, 720),
            jpeg_quality=85,
        )

        call_kwargs = mock_stream.call_args[1]
        assert call_kwargs["fps"] == 15
        assert call_kwargs["resolution"] == (1280, 720)
        assert call_kwargs["jpeg_quality"] == 85


def test_vlm_provider_ws_client_callback():
    """Test that ws_client.send_message is used as frame callback."""
    with (
        patch("providers.unitree_realsense_dev_vlm_provider.ws.Client") as mock_ws,
        patch(
            "providers.unitree_realsense_dev_vlm_provider.UnitreeRealSenseDevVideoStream"
        ) as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        UnitreeRealSenseDevVLMProvider(ws_url="ws://localhost:8000")

        call_args = mock_stream.call_args[0]
        assert call_args[0] == mock_ws_instance.send_message
