from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.vlm_gemini_provider import VLMGeminiProvider


@pytest.fixture
def base_url():
    return "https://api.openmind.com/api/core/gemini"


@pytest.fixture
def fps():
    return 30


@pytest.fixture
def api_key():
    return "test_api_key"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    VLMGeminiProvider.reset()  # type: ignore
    yield
    VLMGeminiProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    mock_client_instance = MagicMock()
    mock_video_stream_instance = MagicMock()
    with (
        patch(
            "providers.vlm_gemini_provider.AsyncOpenAI",
            return_value=mock_client_instance,
        ) as mock_client_class,
        patch(
            "providers.vlm_gemini_provider.VideoStream",
            return_value=mock_video_stream_instance,
        ) as mock_video_stream_class,
    ):
        yield mock_client_class, mock_video_stream_class, mock_client_instance, mock_video_stream_instance


def test_initialization(base_url, api_key, fps, mock_dependencies):
    (
        mock_client_class,
        mock_video_stream_class,
        mock_client_instance,
        mock_video_stream_instance,
    ) = mock_dependencies
    provider = VLMGeminiProvider(base_url, api_key, fps=fps)

    mock_client_class.assert_called_once_with(api_key=api_key, base_url=base_url)
    mock_video_stream_class.assert_called_once_with(frame_callback=provider._process_frame, fps=fps, device_index=0)

    assert not provider.running
    assert provider.api_client is mock_client_instance
    assert provider.video_stream is mock_video_stream_instance


def test_singleton_pattern(base_url, api_key, fps, mock_dependencies):
    provider1 = VLMGeminiProvider(base_url, api_key, fps=fps)
    provider2 = VLMGeminiProvider(base_url, api_key, fps=fps)

    assert provider1 is provider2
    assert provider1.api_client is provider2.api_client
    assert provider1.video_stream is provider2.video_stream


def test_register_message_callback(base_url, api_key, fps, mock_dependencies):
    provider = VLMGeminiProvider(base_url, api_key, fps=fps)
    callback = MagicMock()

    provider.register_message_callback(callback)
    assert provider.message_callback == callback


@pytest.mark.asyncio
async def test_start(base_url, api_key, fps, mock_dependencies):
    (
        _,
        _,
        mock_client_instance,
        mock_video_stream_instance,
    ) = mock_dependencies
    provider = VLMGeminiProvider(base_url, api_key, fps=fps)
    provider.start()

    assert provider.running
    mock_video_stream_instance.start.assert_called_once()

    # The provider unwraps the raw OpenAI HTTP response with `with_raw_response.create`
    # and then parses `raw.text` as JSON. Wire the mock to return that shape so the
    # callback path exercises end-to-end.
    raw_response = MagicMock()
    raw_response.text = '{"choices": [{"message": {"content": "ok"}}]}'
    mock_client_instance.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)

    # Simulate processing a frame so the async API call is triggered.
    await provider._process_frame("fake_frame")
    mock_client_instance.chat.completions.with_raw_response.create.assert_called_once()


def test_stop(base_url, api_key, fps, mock_dependencies):
    (
        _,
        _,
        _,
        mock_video_stream_instance,
    ) = mock_dependencies
    provider = VLMGeminiProvider(base_url, api_key, fps=fps)
    provider.start()
    provider.stop()

    assert not provider.running
    mock_video_stream_instance.stop.assert_called_once()
