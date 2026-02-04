from queue import Queue
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.whisper_asr import WhisperASRInput, WhisperASRSensorConfig


@pytest.fixture
def mock_all_providers():
    with (
        patch("inputs.plugins.whisper_asr.IOProvider"),
        patch("inputs.plugins.whisper_asr.WhisperASRProvider") as mock_asr,
        patch("inputs.plugins.whisper_asr.SleepTickerProvider"),
        patch("inputs.plugins.whisper_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.whisper_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance
        yield mock_asr, mock_asr_instance


def test_initialization(mock_all_providers):
    mock_asr, mock_asr_instance = mock_all_providers
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    assert hasattr(sensor, "messages")
    assert isinstance(sensor.message_buffer, Queue)
    mock_asr_instance.start.assert_called_once()
    mock_asr_instance.register_message_callback.assert_called_once()


def test_initialization_custom_config(mock_all_providers):
    mock_asr, _ = mock_all_providers
    config = WhisperASRSensorConfig(
        model_size="small",
        device="cpu",
        compute_type="int8",
        language="tr",
    )
    WhisperASRInput(config=config)

    mock_asr.assert_called_once_with(
        model_size="small",
        device="cpu",
        compute_type="int8",
        language="tr",
        device_id=None,
        microphone_name=None,
        rate=16000,
        chunk=3200,
        enable_tts_interrupt=False,
    )


@pytest.mark.asyncio
async def test_poll_with_message(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)
    sensor.message_buffer.put("Test speech from Whisper")

    with patch("inputs.plugins.whisper_asr.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert result == "Test speech from Whisper"


@pytest.mark.asyncio
async def test_poll_empty(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    with patch("inputs.plugins.whisper_asr.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert result is None


def test_handle_asr_message(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    sensor._handle_asr_message('{"asr_reply": "hello world test"}')

    assert not sensor.message_buffer.empty()
    assert sensor.message_buffer.get() == "hello world test"


def test_handle_asr_message_single_word_skipped(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    sensor._handle_asr_message('{"asr_reply": "hello"}')

    assert sensor.message_buffer.empty()


def test_handle_asr_message_invalid_json(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    sensor._handle_asr_message("not valid json")

    assert sensor.message_buffer.empty()


def test_handle_asr_message_no_asr_reply_key(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    sensor._handle_asr_message('{"other_key": "value"}')

    assert sensor.message_buffer.empty()


def test_formatted_latest_buffer_empty(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    result = sensor.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    sensor.messages = []
    sensor.messages.append("hello world how are you")

    result = sensor.formatted_latest_buffer()
    assert isinstance(result, str)
    assert "INPUT:" in result
    assert "Voice" in result
    assert "hello world how are you" in result
    assert "// START" in result
    assert "// END" in result
    assert len(sensor.messages) == 0


@pytest.mark.asyncio
async def test_raw_to_text(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    result = await sensor._raw_to_text(None)
    assert result is None

    result = await sensor._raw_to_text("test message")
    assert isinstance(result, Message)
    assert result.message == "test message"


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management(mock_all_providers):
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    await sensor.raw_to_text("first message")
    assert len(sensor.messages) == 1
    assert sensor.messages[0] == "first message"

    await sensor.raw_to_text("second message")
    assert len(sensor.messages) == 1
    assert sensor.messages[0] == "first message second message"


def test_stop(mock_all_providers):
    mock_asr, mock_asr_instance = mock_all_providers
    config = WhisperASRSensorConfig()
    sensor = WhisperASRInput(config=config)

    sensor.stop()
    mock_asr_instance.stop.assert_called_once()


def test_default_config_values():
    config = WhisperASRSensorConfig()
    assert config.model_size == "turbo"
    assert config.device == "auto"
    assert config.compute_type == "auto"
    assert config.language is None
    assert config.rate == 16000
    assert config.chunk == 3200
    assert config.enable_tts_interrupt is False
