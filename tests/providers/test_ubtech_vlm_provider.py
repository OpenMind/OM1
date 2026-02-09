from unittest.mock import MagicMock, patch

import pytest

from providers.ubtech_vlm_provider import UbtechVLMProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UbtechVLMProvider.reset()  # type: ignore
    yield

    try:
        provider = getattr(UbtechVLMProvider, "_instance", None)
        if provider:
            provider.stop()
    except Exception:
        pass

    UbtechVLMProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for UbtechVLMProvider."""
    with (
        patch("providers.ubtech_vlm_provider.ws.Client") as mock_ws_client,
        patch(
            "providers.ubtech_vlm_provider.UbtechCameraVideoStream"
        ) as mock_video_stream,
    ):

        mock_ws_instance = MagicMock()
        mock_ws_client.return_value = mock_ws_instance

        mock_stream_instance = MagicMock()
        mock_video_stream.return_value = mock_stream_instance

        yield {
            "ws_client": mock_ws_client,
            "ws_instance": mock_ws_instance,
            "video_stream": mock_video_stream,
            "stream_instance": mock_stream_instance,
        }


def test_initialization(mock_dependencies):
    """Test UbtechVLMProvider initialization."""
    provider = UbtechVLMProvider(
        ws_url="ws://localhost:8000",
        robot_ip="192.168.1.100",
        fps=30,
        resolution=(640, 480),
        jpeg_quality=70,
    )

    assert provider.robot_ip == "192.168.1.100"
    assert provider.running is False

    mock_dependencies["ws_client"].assert_called_with(url="ws://localhost:8000")
    mock_dependencies["video_stream"].assert_called_once()


def test_initialization_with_stream_url(mock_dependencies):
    """Test initialization with stream URL."""
    UbtechVLMProvider(
        ws_url="ws://localhost:8000",
        robot_ip="192.168.1.100",
        stream_url="ws://localhost:8001",
    )

    assert mock_dependencies["ws_client"].call_count == 2


def test_initialization_without_stream_url(mock_dependencies):
    """Test initialization without stream URL."""
    provider = UbtechVLMProvider(ws_url="ws://localhost:8000", robot_ip="192.168.1.100")

    assert provider.stream_ws_client is None


def test_singleton_pattern(mock_dependencies):
    """Test that UbtechVLMProvider follows singleton pattern."""
    provider1 = UbtechVLMProvider(
        ws_url="ws://localhost:8000", robot_ip="192.168.1.100"
    )
    provider2 = UbtechVLMProvider(
        ws_url="ws://localhost:8001", robot_ip="192.168.1.101"
    )
    assert provider1 is provider2


def test_register_message_callback(mock_dependencies):
    """Test registering message callback."""
    provider = UbtechVLMProvider(ws_url="ws://localhost:8000", robot_ip="192.168.1.100")

    def callback(message):
        pass

    provider.register_message_callback(callback)

    mock_dependencies["ws_instance"].register_message_callback.assert_called_once_with(
        callback
    )


def test_register_message_callback_none(mock_dependencies):
    """Test registering None callback."""
    provider = UbtechVLMProvider(ws_url="ws://localhost:8000", robot_ip="192.168.1.100")

    provider.register_message_callback(None)

    mock_dependencies["ws_instance"].register_message_callback.assert_not_called()


def test_start(mock_dependencies):
    """Test starting the VLM provider."""
    provider = UbtechVLMProvider(ws_url="ws://localhost:8000", robot_ip="192.168.1.100")

    provider.start()

    assert provider.running is True
    mock_dependencies["ws_instance"].start.assert_called_once()
    mock_dependencies["stream_instance"].start.assert_called_once()


def test_start_with_stream_ws(mock_dependencies):
    """Test starting with stream WebSocket client."""
    mock_ws_instance1 = MagicMock()
    mock_ws_instance2 = MagicMock()
    mock_dependencies["ws_client"].side_effect = [mock_ws_instance1, mock_ws_instance2]

    provider = UbtechVLMProvider(
        ws_url="ws://localhost:8000",
        robot_ip="192.168.1.100",
        stream_url="ws://localhost:8001",
    )

    provider.start()

    assert provider.running is True
    mock_ws_instance1.start.assert_called_once()
    mock_ws_instance2.start.assert_called_once()
    mock_dependencies["stream_instance"].register_frame_callback.assert_called_once()


def test_start_already_running(mock_dependencies):
    """Test starting when already running."""
    provider = UbtechVLMProvider(ws_url="ws://localhost:8000", robot_ip="192.168.1.100")

    provider.start()

    mock_dependencies["ws_instance"].start.reset_mock()
    mock_dependencies["stream_instance"].start.reset_mock()

    provider.start()

    mock_dependencies["ws_instance"].start.assert_not_called()
    mock_dependencies["stream_instance"].start.assert_not_called()


def test_stop(mock_dependencies):
    """Test stopping the VLM provider."""
    provider = UbtechVLMProvider(ws_url="ws://localhost:8000", robot_ip="192.168.1.100")

    provider.start()
    provider.stop()

    assert provider.running is False
    mock_dependencies["stream_instance"].stop.assert_called_once()
    mock_dependencies["ws_instance"].stop.assert_called_once()


def test_stop_with_stream_ws(mock_dependencies):
    """Test stopping with stream WebSocket client."""
    mock_ws_instance1 = MagicMock()
    mock_ws_instance2 = MagicMock()
    mock_dependencies["ws_client"].side_effect = [mock_ws_instance1, mock_ws_instance2]

    provider = UbtechVLMProvider(
        ws_url="ws://localhost:8000",
        robot_ip="192.168.1.100",
        stream_url="ws://localhost:8001",
    )

    provider.start()
    provider.stop()

    assert provider.running is False
    mock_dependencies["stream_instance"].stop.assert_called_once()
    mock_ws_instance1.stop.assert_called_once()
    mock_ws_instance2.stop.assert_called_once()


def test_video_stream_parameters(mock_dependencies):
    """Test that video stream is created with correct parameters."""
    UbtechVLMProvider(
        ws_url="ws://localhost:8000",
        robot_ip="192.168.1.100",
        fps=15,
        resolution=(1280, 720),
        jpeg_quality=85,
    )

    call_kwargs = mock_dependencies["video_stream"].call_args[1]
    assert call_kwargs["fps"] == 15
    assert call_kwargs["resolution"] == (1280, 720)
    assert call_kwargs["jpeg_quality"] == 85
    assert call_kwargs["robot_ip"] == "192.168.1.100"
