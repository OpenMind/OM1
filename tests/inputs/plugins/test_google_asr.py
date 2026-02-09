from queue import Queue
from unittest.mock import AsyncMock, patch

import pytest

from inputs.plugins.google_asr import GoogleASRInput, GoogleASRSensorConfig


def test_initialization():
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        assert hasattr(sensor, "messages")
        assert isinstance(sensor.message_buffer, Queue)


@pytest.mark.asyncio
async def test_poll_with_message():
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.message_buffer.put("Test speech")

        with patch("inputs.plugins.google_asr.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result == "Test speech"


def test_formatted_latest_buffer():
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        # Just add the message string, not the Message object
        test_message_content = "hello world how are you"
        sensor.messages = []
        sensor.messages.append(test_message_content)

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Voice" in result
        assert "hello world how are you" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0


def test_resolve_audio_device_auto_detection():
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.pyaudio.PyAudio") as mock_pyaudio,
    ):
        mock_audio_instance = mock_pyaudio.return_value
        mock_audio_instance.get_default_input_device_info.return_value = {
            "index": 1,
            "name": "Default Microphone",
        }

        sensor = GoogleASRInput.__new__(GoogleASRInput)

        result = sensor._resolve_audio_device(None)
        assert result == 1


def test_resolve_audio_device_explicit():
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        sensor = GoogleASRInput.__new__(GoogleASRInput)

        result = sensor._resolve_audio_device(2)
        assert result == 2


def test_resolve_audio_device_string():
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        sensor = GoogleASRInput.__new__(GoogleASRInput)

        result = sensor._resolve_audio_device("hw:1,0")
        assert result == "hw:1,0"


def test_resolve_audio_device_error_fallback():
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.pyaudio.PyAudio") as mock_pyaudio,
    ):
        mock_audio_instance = mock_pyaudio.return_value
        mock_audio_instance.get_default_input_device_info.side_effect = Exception(
            "No device"
        )

        sensor = GoogleASRInput.__new__(GoogleASRInput)

        result = sensor._resolve_audio_device(None)
        assert result is None


def test_config_microphone_device_type():
    config_int = GoogleASRSensorConfig(microphone_device_id=0)
    assert config_int.microphone_device_id == 0

    config_str = GoogleASRSensorConfig(microphone_device_id="hw:1,0")
    assert config_str.microphone_device_id == "hw:1,0"

    config_none = GoogleASRSensorConfig()
    assert config_none.microphone_device_id is None
