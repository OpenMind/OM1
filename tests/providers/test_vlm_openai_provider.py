from unittest.mock import Mock, patch

import pytest

from providers.vlm_openai_provider import VLMOpenAIProvider


@pytest.fixture
def base_url():
    return "https://api.openmind.org/api/core/openai"


@pytest.fixture
def fps():
    return 30


@pytest.fixture
def api_key():
    return "test_api_key"


@pytest.fixture(autouse=True)
def reset_singleton():
    VLMOpenAIProvider._instance = None
    yield


@pytest.fixture
def mock_dependencies():
    with (
        patch("providers.vlm_openai_provider.AsyncOpenAI") as mock_client,
        patch("providers.vlm_openai_provider.VideoStream") as mock_video_stream,
    ):
        yield mock_client, mock_video_stream


def test_initialization(base_url, api_key, fps, mock_dependencies):
    mock_client, mock_video_stream = mock_dependencies
    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)

    mock_client.assert_called_once_with(api_key=api_key, base_url=base_url)
    mock_video_stream.assert_called_once_with(
        frame_callback=provider._process_frame, fps=fps, device_index=0
    )

    assert not provider.running
    assert provider.api_client is not None
    assert provider.video_stream is not None


def test_singleton_pattern(base_url, api_key, fps):
    provider1 = VLMOpenAIProvider(base_url, api_key, fps=fps)
    provider2 = VLMOpenAIProvider(base_url, api_key, fps=fps)

    assert provider1 is provider2
    assert provider1.api_client is provider2.api_client
    assert provider1.video_stream is provider2.video_stream


def test_register_message_callback(base_url, api_key, fps, mock_dependencies):
    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)
    callback = Mock()

    provider.register_message_callback(callback)
    assert provider.message_callback == callback


@pytest.mark.asyncio
async def test_start(base_url, api_key, fps, mock_dependencies):
    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)
    provider.start()

    assert provider.running
    provider.video_stream.start.assert_called_once()

    # Simulate processing a frame so the async API call is triggered.
    # (Using "fake_frame" as an example frame.)
    await provider._process_frame("fake_frame")
    # Now assert the chat.completions.create was called.
    provider.api_client.chat.completions.create.assert_called_once()


def test_stop(base_url, api_key, fps, mock_dependencies):
    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)
    provider.start()
    provider.stop()

    assert not provider.running
    provider.video_stream.stop.assert_called_once()
