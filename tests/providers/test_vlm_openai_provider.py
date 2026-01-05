from unittest.mock import MagicMock, patch, AsyncMock

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
    VLMOpenAIProvider.reset()  # type: ignore
    yield
    VLMOpenAIProvider.reset()  # type: ignore


@pytest.fixture
def mock_dependencies():
    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create = AsyncMock()

    mock_video_stream_instance = MagicMock()

    with (
        patch(
            "providers.vlm_openai_provider.AsyncOpenAI",
            return_value=mock_client_instance,
        ) as mock_client_class,
        patch(
            "providers.vlm_openai_provider.VideoStream",
            return_value=mock_video_stream_instance,
        ) as mock_video_stream_class,
    ):
        yield (
            mock_client_class,
            mock_video_stream_class,
            mock_client_instance,
            mock_video_stream_instance,
        )


def test_initialization(base_url, api_key, fps, mock_dependencies):
    (
        mock_client_class,
        mock_video_stream_class,
        mock_client_instance,
        mock_video_stream_instance,
    ) = mock_dependencies

    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)

    mock_client_class.assert_called_once_with(
        api_key=api_key,
        base_url=base_url,
    )

    mock_video_stream_class.assert_called_once_with(
        frame_callback=provider._process_frame,
        fps=fps,
        device_index=0,
    )

    assert provider.running is False
    assert provider.api_client is mock_client_instance
    assert provider.video_stream is mock_video_stream_instance


def test_singleton_pattern(base_url, api_key, fps, mock_dependencies):
    provider1 = VLMOpenAIProvider(base_url, api_key, fps=fps)
    provider2 = VLMOpenAIProvider(base_url, api_key, fps=fps)

    assert provider1 is provider2
    assert provider1.api_client is provider2.api_client
    assert provider1.video_stream is provider2.video_stream


def test_register_message_callback(base_url, api_key, fps, mock_dependencies):
    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)
    callback = MagicMock()

    provider.register_message_callback(callback)

    assert provider.message_callback is callback


@pytest.mark.asyncio
async def test_process_frame_when_running(base_url, api_key, fps, mock_dependencies):
    _, _, mock_client_instance, _ = mock_dependencies

    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)
    provider.start()

    await provider._process_frame("fake_frame")

    mock_client_instance.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_process_frame_when_stopped(base_url, api_key, fps, mock_dependencies):
    _, _, mock_client_instance, _ = mock_dependencies

    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)

    await provider._process_frame("fake_frame")

    mock_client_instance.chat.completions.create.assert_not_called()


def test_start(base_url, api_key, fps, mock_dependencies):
    _, _, _, mock_video_stream_instance = mock_dependencies

    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)
    provider.start()

    assert provider.running is True
    mock_video_stream_instance.start.assert_called_once()


def test_stop(base_url, api_key, fps, mock_dependencies):
    _, _, _, mock_video_stream_instance = mock_dependencies

    provider = VLMOpenAIProvider(base_url, api_key, fps=fps)
    provider.start()
    provider.stop()

    assert provider.running is False
    mock_video_stream_instance.stop.assert_called_once()
