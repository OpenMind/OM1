from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.vlm_gemini_zenoh_provider import VLMGeminiZenohProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    VLMGeminiZenohProvider.reset()  # type: ignore
    yield
    VLMGeminiZenohProvider.reset()  # type: ignore


@pytest.fixture
def deps():
    with (
        patch("providers.vlm_gemini_zenoh_provider.AsyncOpenAI") as mock_openai_class,
        patch("providers.vlm_gemini_zenoh_provider.VideoZenohStream") as mock_video_class,
        patch("providers.vlm_gemini_zenoh_provider.threading.Thread") as mock_thread_class,
    ):
        client = MagicMock()
        mock_openai_class.return_value = client
        video = MagicMock()
        mock_video_class.return_value = video
        thread = MagicMock()
        mock_thread_class.return_value = thread
        yield {
            "openai_class": mock_openai_class,
            "client": client,
            "video_class": mock_video_class,
            "video": video,
            "thread_class": mock_thread_class,
            "thread": thread,
        }


def test_initialization(deps):
    provider = VLMGeminiZenohProvider(base_url="http://x", api_key="k", topic="rgb_image")
    assert provider.running is False
    assert provider.model == "gemini-2.5-flash"
    assert provider.max_tokens == 1024
    assert provider.message_callback is None
    deps["openai_class"].assert_called_once_with(api_key="k", base_url="http://x")
    deps["video_class"].assert_called_once()
    deps["thread"].start.assert_called_once()


def test_initialization_custom_model_prompt(deps):
    provider = VLMGeminiZenohProvider(
        base_url="http://x",
        api_key="k",
        model="gemini-2.5-pro",
        max_tokens=512,
        prompt="describe",
    )
    assert provider.model == "gemini-2.5-pro"
    assert provider.max_tokens == 512
    assert provider.prompt == "describe"


def test_register_message_callback(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    cb = MagicMock()
    provider.register_message_callback(cb)
    assert provider.message_callback is cb


def test_start_idempotent(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    provider.start()
    assert provider.running is True
    deps["video"].start.assert_called_once()
    # Second call should warn but not re-start.
    provider.start()
    deps["video"].start.assert_called_once()


def test_stop(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    provider.start()
    # Give the loop a no-op stop method
    provider._loop = MagicMock()
    provider.stop()
    assert provider.running is False
    deps["video"].stop.assert_called_once()
    provider._loop.call_soon_threadsafe.assert_called_once()


def test_dispatch_frame_backpressure(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    provider._loop = MagicMock()
    provider._max_inflight = 1
    provider._inflight = 1  # already at limit
    with patch("providers.vlm_gemini_zenoh_provider.asyncio.run_coroutine_threadsafe") as mock_schedule:
        provider._dispatch_frame("frame")
        mock_schedule.assert_not_called()


def test_dispatch_frame_schedules(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    provider._loop = MagicMock()
    with patch("providers.vlm_gemini_zenoh_provider.asyncio.run_coroutine_threadsafe") as mock_schedule:
        # Need to consume the awaitable created by _process_frame
        async def fake_process(_frame):
            return None

        with patch.object(provider, "_process_frame", side_effect=fake_process):
            provider._dispatch_frame("frame")
        mock_schedule.assert_called_once()
        assert provider._inflight == 1


@pytest.mark.asyncio
async def test_process_frame_success_calls_callback(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    raw = MagicMock()
    raw.text = '{"choices": [{"message": {"content": "ok"}}]}'
    provider.api_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw)
    cb = MagicMock()
    provider.register_message_callback(cb)
    provider._inflight = 1
    await provider._process_frame('{"frame": "BASE64", "timestamp": 0.0}')
    cb.assert_called_once_with("ok")
    assert provider._inflight == 0


@pytest.mark.asyncio
async def test_process_frame_handles_bare_b64(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    raw = MagicMock()
    raw.text = '{"choices": [{"message": {"content": "fine"}}]}'
    provider.api_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw)
    cb = MagicMock()
    provider.register_message_callback(cb)
    provider._inflight = 1
    await provider._process_frame("not-json-but-base64")
    cb.assert_called_once_with("fine")


@pytest.mark.asyncio
async def test_process_frame_swallows_api_errors(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    provider.api_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=RuntimeError("fail"))
    cb = MagicMock()
    provider.register_message_callback(cb)
    provider._inflight = 1
    await provider._process_frame('{"frame": "x"}')
    cb.assert_not_called()
    assert provider._inflight == 0


@pytest.mark.asyncio
async def test_process_frame_no_callback_no_crash(deps):
    provider = VLMGeminiZenohProvider("http://x", "k")
    raw = MagicMock()
    raw.text = '{"choices": [{"message": {"content": "ok"}}]}'
    provider.api_client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw)
    provider._inflight = 1
    # No callback registered — should not raise
    await provider._process_frame('{"frame": "x"}')
