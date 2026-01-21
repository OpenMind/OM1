import json
from unittest.mock import MagicMock, patch

import pytest

from providers.vlm_openai_rtsp_provider import VLMOpenAIRTSPProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    VLMOpenAIRTSPProvider.reset()  # type: ignore
    yield

    try:
        provider = getattr(VLMOpenAIRTSPProvider, "_instance", None)
        if provider:
            provider.running = False
            if hasattr(provider, "batch_task") and provider.batch_task:
                mock_task = MagicMock()
                mock_task.done.return_value = True
                provider.batch_task = mock_task
    except Exception:
        pass

    VLMOpenAIRTSPProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for VLMOpenAIRTSPProvider."""
    with (
        patch(
            "providers.vlm_openai_rtsp_provider.AsyncOpenAI", new_callable=MagicMock
        ) as mock_openai,
        patch("providers.vlm_openai_rtsp_provider.VideoRTSPStream") as mock_stream,
    ):

        mock_client = MagicMock(spec=[])
        mock_openai.return_value = mock_client

        mock_stream_instance = MagicMock()
        mock_stream.return_value = mock_stream_instance

        yield {
            "openai": mock_openai,
            "client": mock_client,
            "stream": mock_stream,
            "stream_instance": mock_stream_instance,
        }


def test_initialization(mock_dependencies):
    """Test VLMOpenAIRTSPProvider initialization."""
    provider = VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000",
        api_key="test-key",
        rtsp_url="rtsp://localhost:8554/camera",
        fps=30,
        batch_size=5,
    )

    assert provider.running is False
    assert (
        provider.prompt
        == "What is the most interesting aspect in this series of images?"
    )
    assert provider.batch_size == 5
    assert provider.batch_interval == 0.5

    mock_dependencies["openai"].assert_called_once()
    mock_dependencies["stream"].assert_called_once()


def test_initialization_with_custom_prompt(mock_dependencies):
    """Test initialization with custom prompt."""
    custom_prompt = "Describe what you see"

    provider = VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000", api_key="test-key", prompt=custom_prompt
    )

    assert provider.prompt == custom_prompt


def test_initialization_with_custom_batch_settings(mock_dependencies):
    """Test initialization with custom batch settings."""
    provider = VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000",
        api_key="test-key",
        batch_size=10,
        batch_interval=1.0,
    )

    assert provider.batch_size == 10
    assert provider.batch_interval == 1.0
    assert provider.frame_queue.maxlen == 10


def test_singleton_pattern(mock_dependencies):
    """Test that VLMOpenAIRTSPProvider follows singleton pattern."""
    provider1 = VLMOpenAIRTSPProvider(base_url="http://localhost:8000", api_key="key1")
    provider2 = VLMOpenAIRTSPProvider(base_url="http://localhost:9000", api_key="key2")
    assert provider1 is provider2


def test_queue_frame(mock_dependencies):
    """Test queuing a frame."""
    provider = VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000", api_key="test-key"
    )

    frame_data = json.dumps({"frame": "base64encodeddata"})
    provider._queue_frame(frame_data)

    assert len(provider.frame_queue) == 1
    assert provider.frame_queue[0] == "base64encodeddata"


def test_queue_frame_with_invalid_json(mock_dependencies):
    """Test queuing a frame with invalid JSON."""
    provider = VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000", api_key="test-key"
    )

    # Should not raise exception, just log error
    provider._queue_frame("invalid json")

    assert len(provider.frame_queue) == 0


def test_frame_queue_maxlen(mock_dependencies):
    """Test that frame queue respects maxlen."""
    provider = VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000", api_key="test-key", batch_size=3
    )

    # Add more frames than batch_size
    for i in range(5):
        frame_data = json.dumps({"frame": f"frame{i}"})
        provider._queue_frame(frame_data)

    # Should only keep last 3
    assert len(provider.frame_queue) == 3


def test_register_message_callback(mock_dependencies):
    """Test registering message callback."""
    provider = VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000", api_key="test-key"
    )

    def callback(message):
        pass

    provider.register_message_callback(callback)

    assert provider.message_callback == callback


def test_video_stream_parameters(mock_dependencies):
    """Test that video stream is created with correct parameters."""
    VLMOpenAIRTSPProvider(
        base_url="http://localhost:8000",
        api_key="test-key",
        rtsp_url="rtsp://test:8554/cam",
        decode_format="H265",
        fps=15,
    )

    call_args = mock_dependencies["stream"].call_args
    assert call_args[0][0] == "rtsp://test:8554/cam"
    assert call_args[0][1] == "H265"
    assert call_args[1]["fps"] == 15


def test_openai_client_initialization(mock_dependencies):
    """Test that AsyncOpenAI client is initialized correctly."""
    VLMOpenAIRTSPProvider(base_url="http://localhost:8000", api_key="test-key-123")

    call_kwargs = mock_dependencies["openai"].call_args[1]
    assert call_kwargs["api_key"] == "test-key-123"
    assert call_kwargs["base_url"] == "http://localhost:8000"
