from unittest.mock import MagicMock, patch

import pytest

from providers.vlm_vila_rtsp_provider import VLMVilaRTSPProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    VLMVilaRTSPProvider.reset()  # type: ignore
    yield

    try:
        provider = getattr(VLMVilaRTSPProvider, "_instance", None)
        if provider:
            provider.stop()
    except Exception:
        pass

    VLMVilaRTSPProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for VLMVilaRTSPProvider."""
    with (
        patch("providers.vlm_vila_rtsp_provider.ws.Client") as mock_ws,
        patch("providers.vlm_vila_rtsp_provider.VideoRTSPStream") as mock_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        yield {
            "ws": mock_ws,
            "ws_instance": mock_ws_instance,
            "stream": mock_stream,
            "stream_instance": mock_stream_instance,
        }


def test_initialization(mock_dependencies):
    """Test VLMVilaRTSPProvider initialization."""
    provider = VLMVilaRTSPProvider(
        ws_url="ws://localhost:8000",
        rtsp_url="rtsp://localhost:8554/camera",
        decode_format="H264",
        fps=30,
    )

    assert provider.running is False

    mock_dependencies["ws"].assert_called_once_with(url="ws://localhost:8000")
    mock_dependencies["stream"].assert_called_once()


def test_initialization_defaults(mock_dependencies):
    """Test initialization with default values."""
    VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][0] == "rtsp://localhost:8554/top_camera"
    assert call_args[0][1] == "H264"
    assert call_args[1]["fps"] == 30


def test_singleton_pattern(mock_dependencies):
    """Test that VLMVilaRTSPProvider follows singleton pattern."""
    provider1 = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")
    provider2 = VLMVilaRTSPProvider(ws_url="ws://localhost:9000")
    assert provider1 is provider2


def test_register_frame_callback(mock_dependencies):
    """Test registering frame callback."""
    provider = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    def callback(frame):
        pass

    provider.register_frame_callback(callback)

    mock_dependencies[
        "stream_instance"
    ].register_frame_callback.assert_called_once_with(callback)


def test_register_frame_callback_none(mock_dependencies):
    """Test registering None frame callback."""
    provider = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    provider.register_frame_callback(None)

    mock_dependencies["stream_instance"].register_frame_callback.assert_not_called()


def test_register_message_callback(mock_dependencies):
    """Test registering message callback."""
    provider = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    def callback(message):
        pass

    provider.register_message_callback(callback)

    mock_dependencies["ws_instance"].register_message_callback.assert_called_once_with(
        callback
    )


def test_register_message_callback_none(mock_dependencies):
    """Test registering None message callback."""
    provider = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    provider.register_message_callback(None)

    mock_dependencies["ws_instance"].register_message_callback.assert_not_called()


def test_start(mock_dependencies):
    """Test starting the provider."""
    provider = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    provider.start()

    assert provider.running is True
    mock_dependencies["ws_instance"].start.assert_called_once()
    mock_dependencies["stream_instance"].start.assert_called_once()


def test_start_already_running(mock_dependencies):
    """Test starting when already running."""
    provider = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    provider.start()

    # Reset mocks
    mock_dependencies["ws_instance"].start.reset_mock()
    mock_dependencies["stream_instance"].start.reset_mock()

    # Try to start again
    provider.start()

    # Should not call start again
    mock_dependencies["ws_instance"].start.assert_not_called()
    mock_dependencies["stream_instance"].start.assert_not_called()


def test_stop(mock_dependencies):
    """Test stopping the provider."""
    provider = VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    provider.start()
    provider.stop()

    assert provider.running is False
    mock_dependencies["stream_instance"].stop.assert_called_once()
    mock_dependencies["ws_instance"].stop.assert_called_once()


def test_custom_rtsp_url(mock_dependencies):
    """Test custom RTSP URL."""
    custom_url = "rtsp://10.0.0.100:8554/custom_camera"

    VLMVilaRTSPProvider(ws_url="ws://localhost:8000", rtsp_url=custom_url)

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][0] == custom_url


def test_custom_decode_format(mock_dependencies):
    """Test custom decode format."""
    VLMVilaRTSPProvider(ws_url="ws://localhost:8000", decode_format="H265")

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][1] == "H265"


def test_custom_fps(mock_dependencies):
    """Test custom FPS."""
    VLMVilaRTSPProvider(ws_url="ws://localhost:8000", fps=15)

    call_args = mock_dependencies["stream"].call_args
    assert call_args[1]["fps"] == 15


def test_ws_client_send_message_callback(mock_dependencies):
    """Test that ws_client.send_message is used as frame callback."""
    VLMVilaRTSPProvider(ws_url="ws://localhost:8000")

    call_args = mock_dependencies["stream"].call_args
    assert (
        call_args[1]["frame_callback"] == mock_dependencies["ws_instance"].send_message
    )
