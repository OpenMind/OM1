from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.vlm_gemini_zenoh import VLMGeminiZenoh, VLMGeminiZenohConfig


@pytest.fixture
def patches(monkeypatch):
    monkeypatch.setenv("OM_API_KEY", "env-key")
    with (
        patch("inputs.plugins.vlm_gemini_zenoh.IOProvider"),
        patch("inputs.plugins.vlm_gemini_zenoh.VLMGeminiZenohProvider") as mock_provider_class,
    ):
        instance = MagicMock()
        mock_provider_class.return_value = instance
        yield {"provider_class": mock_provider_class, "provider": instance}


def test_initialization_with_explicit_api_key(patches, monkeypatch):
    monkeypatch.delenv("OM_API_KEY", raising=False)
    config = VLMGeminiZenohConfig(api_key="explicit")
    sensor = VLMGeminiZenoh(config=config)
    assert sensor.descriptor_for_LLM == "Vision"
    patches["provider"].start.assert_called_once()
    patches["provider"].register_message_callback.assert_called_once()


def test_initialization_falls_back_to_env(patches):
    VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    _args, kwargs = patches["provider_class"].call_args
    assert kwargs["api_key"] == "env-key"
    assert kwargs["topic"] == "rgb_image"
    assert kwargs["model"] == "gemini-2.5-flash"
    # default branch (no explicit prompt) should not pass `prompt` kwarg
    assert "prompt" not in kwargs


def test_initialization_with_custom_prompt(patches):
    config = VLMGeminiZenohConfig(prompt="describe the dog")
    VLMGeminiZenoh(config=config)
    _, kwargs = patches["provider_class"].call_args
    assert kwargs["prompt"] == "describe the dog"


def test_initialization_raises_without_api_key(patches, monkeypatch):
    monkeypatch.delenv("OM_API_KEY", raising=False)
    config = VLMGeminiZenohConfig(api_key=None)
    with pytest.raises(ValueError, match="api_key"):
        VLMGeminiZenoh(config=config)


def test_handle_vlm_message_buffers_content(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    sensor._handle_vlm_message("a banana")
    assert sensor.message_buffer.get_nowait() == "a banana"


def test_handle_vlm_message_drops_empty(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    sensor._handle_vlm_message("")
    assert sensor.message_buffer.empty()


@pytest.mark.asyncio
async def test_poll_returns_buffered_message(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    sensor.message_buffer.put("hello")
    with patch("inputs.plugins.vlm_gemini_zenoh.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()
    assert result == "hello"


@pytest.mark.asyncio
async def test_poll_returns_none_when_empty(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    with patch("inputs.plugins.vlm_gemini_zenoh.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_returns_message(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    msg = await sensor._raw_to_text("a cat")
    assert isinstance(msg, Message)
    assert msg.message == "a cat"


@pytest.mark.asyncio
async def test_raw_to_text_none_returns_none(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    assert await sensor._raw_to_text(None) is None


@pytest.mark.asyncio
async def test_raw_to_text_appends(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    await sensor.raw_to_text("a cat")
    assert len(sensor.messages) == 1


@pytest.mark.asyncio
async def test_raw_to_text_none_input_skips(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    await sensor.raw_to_text(None)
    assert sensor.messages == []


def test_formatted_latest_buffer_empty(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    assert sensor.formatted_latest_buffer() is None


def test_formatted_latest_buffer_with_message(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    sensor.messages.append(Message(timestamp=1.0, message="abstract pixels"))
    result = sensor.formatted_latest_buffer()
    assert result is not None
    assert "INPUT: Vision" in result
    assert "abstract pixels" in result
    assert sensor.messages == []


def test_stop_calls_provider_stop(patches):
    sensor = VLMGeminiZenoh(config=VLMGeminiZenohConfig())
    sensor.stop()
    patches["provider"].stop.assert_called_once()
