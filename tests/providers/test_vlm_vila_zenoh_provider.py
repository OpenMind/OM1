from unittest.mock import MagicMock, patch

import pytest

from providers.vlm_vila_zenoh_provider import VLMVilaZenohProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    VLMVilaZenohProvider.reset()  # type: ignore
    yield

    try:
        provider = getattr(VLMVilaZenohProvider, "_instance", None)
        if provider:
            provider.stop()
    except Exception:
        pass

    VLMVilaZenohProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for VLMVilaZenohProvider."""
    with (
        patch("providers.vlm_vila_zenoh_provider.ws.Client") as mock_ws,
        patch("providers.vlm_vila_zenoh_provider.VideoZenohStream") as mock_stream,
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
    """Test VLMVilaZenohProvider initialization."""
    provider = VLMVilaZenohProvider(
        ws_url="ws://localhost:8000", topic="test/camera", decode_format="H264"
    )

    assert provider.running is False

    mock_dependencies["ws"].assert_called_once_with(url="ws://localhost:8000")
    mock_dependencies["stream"].assert_called_once()


def test_initialization_defaults(mock_dependencies):
    """Test initialization with default values."""
    VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][0] == "rgb_image"
    assert call_args[0][1] == "H264"


def test_singleton_pattern(mock_dependencies):
    """Test that VLMVilaZenohProvider follows singleton pattern."""
    provider1 = VLMVilaZenohProvider(ws_url="ws://localhost:8000")
    provider2 = VLMVilaZenohProvider(ws_url="ws://localhost:9000")
    assert provider1 is provider2


def test_register_frame_callback(mock_dependencies):
    """Test registering frame callback."""
    provider = VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    def callback(frame):
        pass

    provider.register_frame_callback(callback)

    mock_dependencies[
        "stream_instance"
    ].register_frame_callback.assert_called_once_with(callback)


def test_register_frame_callback_none(mock_dependencies):
    """Test registering None frame callback."""
    provider = VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    provider.register_frame_callback(None)

    mock_dependencies["stream_instance"].register_frame_callback.assert_not_called()


def test_register_message_callback(mock_dependencies):
    """Test registering message callback."""
    provider = VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    def callback(message):
        pass

    provider.register_message_callback(callback)

    mock_dependencies["ws_instance"].register_message_callback.assert_called_once_with(
        callback
    )


def test_register_message_callback_none(mock_dependencies):
    """Test registering None message callback."""
    provider = VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    provider.register_message_callback(None)

    mock_dependencies["ws_instance"].register_message_callback.assert_not_called()


def test_start(mock_dependencies):
    """Test starting the provider."""
    provider = VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    provider.start()

    assert provider.running is True
    mock_dependencies["ws_instance"].start.assert_called_once()
    mock_dependencies["stream_instance"].start.assert_called_once()


def test_start_already_running(mock_dependencies):
    """Test starting when already running."""
    provider = VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    provider.start()

    mock_dependencies["ws_instance"].start.reset_mock()
    mock_dependencies["stream_instance"].start.reset_mock()

    provider.start()

    mock_dependencies["ws_instance"].start.assert_not_called()
    mock_dependencies["stream_instance"].start.assert_not_called()


def test_stop(mock_dependencies):
    """Test stopping the provider."""
    provider = VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    provider.start()
    provider.stop()

    assert provider.running is False
    mock_dependencies["stream_instance"].stop.assert_called_once()
    mock_dependencies["ws_instance"].stop.assert_called_once()


def test_custom_topic(mock_dependencies):
    """Test custom topic."""
    custom_topic = "robot/camera/front"

    VLMVilaZenohProvider(ws_url="ws://localhost:8000", topic=custom_topic)

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][0] == custom_topic


def test_custom_decode_format(mock_dependencies):
    """Test custom decode format."""
    VLMVilaZenohProvider(ws_url="ws://localhost:8000", decode_format="H265")

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][1] == "H265"


def test_ws_client_send_message_callback(mock_dependencies):
    """Test that ws_client.send_message is used as frame callback."""
    VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    call_args = mock_dependencies["stream"].call_args
    assert (
        call_args[1]["frame_callback"] == mock_dependencies["ws_instance"].send_message
    )


def test_default_topic_value(mock_dependencies):
    """Test default topic value is 'rgb_image'."""
    VLMVilaZenohProvider(ws_url="ws://localhost:8000")

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][0] == "rgb_image"


def test_video_zenoh_stream_initialization(mock_dependencies):
    """Test VideoZenohStream is initialized correctly."""
    VLMVilaZenohProvider(
        ws_url="ws://localhost:8000", topic="custom_topic", decode_format="H265"
    )

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][0] == "custom_topic"
    assert call_args[0][1] == "H265"
    assert "frame_callback" in call_args[1]
