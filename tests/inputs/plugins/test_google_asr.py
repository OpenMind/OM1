import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.google_asr import GoogleASRInput, GoogleASRSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        assert hasattr(sensor, "messages")
        assert isinstance(sensor.message_buffer, asyncio.Queue)
        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "Voice"
        assert sensor._stopped is False
        mock_asr_instance.start.assert_called_once()
        mock_asr_instance.register_message_callback.assert_called_once()


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(
            api_key="test_key",
            rate=16000,
            chunk=8192,
            base_url="wss://test.com",
            microphone_device_id=1,
            microphone_name="test_mic",
            language="chinese",
            remote_input=True,
            enable_tts_interrupt=True,
        )
        GoogleASRInput(config=config)

        mock_asr.assert_called_with(
            rate=16000,
            chunk=8192,
            ws_url="wss://test.com",
            device_id=1,
            microphone_name="test_mic",
            language_code="cmn-Hans-CN",
            alternative_language_codes=[],
            remote_input=True,
            enable_tts_interrupt=True,
        )


def test_initialization_with_unsupported_language():
    """Test initialization with unsupported language defaults to English."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(language="unsupported_language")
        GoogleASRInput(config=config)

        call_kwargs = mock_asr.call_args[1]
        assert call_kwargs["language_code"] == "en-US"


def test_initialization_zenoh_failure():
    """Test initialization when Zenoh fails."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch(
            "inputs.plugins.google_asr.open_zenoh_session",
            side_effect=Exception("Zenoh failed"),
        ),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        assert sensor.session is None
        assert sensor.asr_publisher is None


@pytest.mark.asyncio
async def test_poll_with_message():
    """Test _poll with message in buffer."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.message_buffer.put_nowait("Test speech")

        result = await sensor._poll()

        assert result == "Test speech"


@pytest.mark.asyncio
async def test_poll_empty_buffer():
    """Test _poll with empty buffer."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is None


def test_handle_asr_message_valid():
    """Test _handle_asr_message with valid message."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        raw_message = json.dumps({"asr_reply": "hello world test"})
        sensor._handle_asr_message(raw_message)

        assert not sensor.message_buffer.empty()
        result = sensor.message_buffer.get_nowait()
        assert result == "hello world test"


def test_handle_asr_message_single_word():
    """Test _handle_asr_message with single word (ignored)."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        raw_message = json.dumps({"asr_reply": "hello"})
        sensor._handle_asr_message(raw_message)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_stopped():
    """Test _handle_asr_message when sensor is stopped."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor._stopped = True

        raw_message = json.dumps({"asr_reply": "hello world"})
        sensor._handle_asr_message(raw_message)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_invalid_json():
    """Test _handle_asr_message with invalid JSON."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        sensor._handle_asr_message("not valid json")

        assert sensor.message_buffer.empty()


def test_handle_asr_message_no_asr_reply():
    """Test _handle_asr_message without asr_reply field."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        raw_message = json.dumps({"other_field": "value"})
        sensor._handle_asr_message(raw_message)

        assert sensor.message_buffer.empty()


@pytest.mark.asyncio
async def test_raw_to_text_none():
    """Test _raw_to_text with None."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_valid():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.time.time", return_value=123.456),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        result = await sensor._raw_to_text("test message")
        assert result is not None
        assert result.message == "test message"
        assert result.timestamp == 123.456


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.google_asr.IOProvider") as mock_io,
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider") as mock_conv,
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_io_instance = MagicMock()
        mock_io.return_value = mock_io_instance
        mock_conv_instance = MagicMock()
        mock_conv.return_value = mock_conv_instance

        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        sensor.messages.append("hello world how are you")

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "Voice" in result
        assert result.count("hello world how are you") == 1
        assert len(sensor.messages) == 0

        mock_io_instance.add_input.assert_called_once()
        mock_io_instance.add_mode_transition_input.assert_called_once()
        mock_conv_instance.store_user_message.assert_called_once()


def test_formatted_latest_buffer_with_zenoh():
    """Test formatted_latest_buffer with Zenoh publishing."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session") as mock_zenoh,
        patch("inputs.plugins.google_asr.ASRText") as mock_asr_text,
        patch("inputs.plugins.google_asr.prepare_header"),
    ):
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        mock_asr_msg = MagicMock()
        mock_asr_msg.serialize.return_value = b"serialized"
        mock_asr_text.return_value = mock_asr_msg

        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.messages.append("test message")

        result = sensor.formatted_latest_buffer()

        assert result is not None
        mock_publisher.put.assert_called_once_with(b"serialized")


def test_formatted_latest_buffer_zenoh_failure():
    """Test formatted_latest_buffer when Zenoh publishing fails."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.put.side_effect = Exception("Zenoh publish failed")
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.messages.append("test message")

        # Should not raise exception
        result = sensor.formatted_latest_buffer()
        assert result is not None


@pytest.mark.asyncio
async def test_raw_to_text_none_skips_sleep_when_buffer_has_messages():
    """Test raw_to_text with None sets skip_sleep when messages exist."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider") as mock_sleep_ticker,
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_sleep_ticker_instance = MagicMock()
        mock_sleep_ticker.return_value = mock_sleep_ticker_instance

        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.messages = ["existing message"]

        await sensor.raw_to_text(None)
        assert mock_sleep_ticker_instance.skip_sleep is True


@pytest.mark.asyncio
async def test_raw_to_text_first_message():
    """Test raw_to_text with first message."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        await sensor.raw_to_text("first message")

        assert len(sensor.messages) == 1
        assert sensor.messages[0] == "first message"


@pytest.mark.asyncio
async def test_raw_to_text_concatenates_messages():
    """Test raw_to_text concatenates when messages already exist."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        await sensor.raw_to_text("hello")
        await sensor.raw_to_text("world")
        assert len(sensor.messages) == 1
        assert sensor.messages[0] == "hello world"


def test_stop():
    """Test stop method stops ASR provider and closes Zenoh session."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        sensor.message_buffer.put_nowait("message 1")
        sensor.message_buffer.put_nowait("message 2")
        sensor.messages.append("message 3")

        sensor.stop()

        assert sensor._stopped is True
        assert sensor.message_buffer.empty()
        assert len(sensor.messages) == 0
        mock_asr_instance.unregister_message_callback.assert_called_once()
        mock_publisher.undeclare.assert_called_once()
        mock_session.close.assert_called_once()


def test_stop_with_exceptions():
    """Test stop method handles exceptions gracefully."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_asr_instance = MagicMock()
        mock_asr_instance.unregister_message_callback.side_effect = Exception("Unregister failed")
        mock_asr_instance.stop.side_effect = Exception("Stop failed")
        mock_asr.return_value = mock_asr_instance

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.undeclare.side_effect = Exception("Undeclare failed")
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True


def test_stop_no_session():
    """Test stop method when session is None."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        sensor.session = None

        sensor.stop()


def test_initialization_with_api_version_v1():
    """Test initialization with API version v1."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(api_version="v1", api_key="test_key")
        GoogleASRInput(config=config)

        call_kwargs = mock_asr.call_args[1]
        assert call_kwargs["ws_url"] == "wss://api.openmind.com/api/core/google/asr/v1?api_key=test_key"


def test_initialization_with_api_version_v2():
    """Test initialization with API version v2."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(api_version="v2", api_key="test_key")
        GoogleASRInput(config=config)

        call_kwargs = mock_asr.call_args[1]
        assert call_kwargs["ws_url"] == "wss://api.openmind.com/api/core/google/asr/v2?api_key=test_key"


def test_initialization_with_invalid_api_version_defaults_to_v2():
    """Test initialization with invalid API version defaults to v2."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(api_version="v3", api_key="test_key")
        GoogleASRInput(config=config)

        call_kwargs = mock_asr.call_args[1]
        assert call_kwargs["ws_url"] == "wss://api.openmind.com/api/core/google/asr/v2?api_key=test_key"


def test_initialization_with_alternative_languages_v1():
    """Test initialization with alternative languages using API v1."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(
            api_version="v1",
            language="english",
            alternative_languages=["chinese", "spanish"],
        )
        GoogleASRInput(config=config)

        call_kwargs = mock_asr.call_args[1]
        assert call_kwargs["language_code"] == "en-US"
        assert "cmn-Hans-CN" in call_kwargs["alternative_language_codes"]
        assert "es-ES" in call_kwargs["alternative_language_codes"]


def test_initialization_with_alternative_languages_v2_ignored():
    """Test initialization with alternative languages using API v2 (should be ignored)."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(
            api_version="v2",
            language="english",
            alternative_languages=["chinese", "spanish"],
        )
        GoogleASRInput(config=config)

        call_kwargs = mock_asr.call_args[1]
        assert call_kwargs["language_code"] == "en-US"
        assert call_kwargs["alternative_language_codes"] == []


def test_initialization_default_api_version_is_v2():
    """Test that default API version is v2."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = GoogleASRSensorConfig(api_key="test_key")
        GoogleASRInput(config=config)

        call_kwargs = mock_asr.call_args[1]
        assert call_kwargs["ws_url"] == "wss://api.openmind.com/api/core/google/asr/v2?api_key=test_key"


def test_handle_asr_message_speech_start_sets_timer():
    """speech_start event sets _speech_start_time."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.time.time", return_value=100.0),
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        assert sensor._speech_start_time is None

        sensor._handle_asr_message(json.dumps({"type": "speech_start"}))

        assert sensor._speech_start_time == 100.0


def test_handle_asr_message_speech_end_records_duration_metrics():
    """speech_end event records speech duration histogram and gauge."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_speech_duration") as mock_hist,
        patch("inputs.plugins.google_asr.om1_asr_speech_duration_last") as mock_gauge,
        patch("inputs.plugins.google_asr.time.time", return_value=105.0),
    ):
        config = GoogleASRSensorConfig(language="english", api_version="v2")
        sensor = GoogleASRInput(config=config)
        sensor._speech_start_time = 100.0

        sensor._handle_asr_message(json.dumps({"type": "speech_end"}))

        mock_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(5.0)
        mock_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(5.0)


def test_handle_asr_message_speech_end_no_op_without_start_time():
    """speech_end event is ignored when _speech_start_time is None."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_speech_duration") as mock_hist,
        patch("inputs.plugins.google_asr.om1_asr_speech_duration_last") as mock_gauge,
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        # _speech_start_time defaults to None

        sensor._handle_asr_message(json.dumps({"type": "speech_end"}))

        mock_hist.labels().observe.assert_not_called()
        mock_gauge.labels().set.assert_not_called()


def test_handle_asr_message_end_of_utterance_records_latency_metrics():
    """end_of_utterance event records utterance latency histogram and gauge."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_utterance_end_latency") as mock_hist,
        patch("inputs.plugins.google_asr.om1_asr_utterance_end_latency_last") as mock_gauge,
        patch("inputs.plugins.google_asr.time.time", return_value=103.5),
    ):
        config = GoogleASRSensorConfig(language="english", api_version="v2")
        sensor = GoogleASRInput(config=config)
        sensor._speech_start_time = 100.0

        sensor._handle_asr_message(json.dumps({"type": "end_of_utterance"}))

        mock_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(3.5)
        mock_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(3.5)


def test_handle_asr_message_end_of_utterance_no_op_without_start_time():
    """end_of_utterance event is ignored when _speech_start_time is None."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_utterance_end_latency") as mock_hist,
        patch("inputs.plugins.google_asr.om1_asr_utterance_end_latency_last") as mock_gauge,
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)

        sensor._handle_asr_message(json.dumps({"type": "end_of_utterance"}))

        mock_hist.labels().observe.assert_not_called()
        mock_gauge.labels().set.assert_not_called()


def test_handle_asr_message_asr_reply_records_latency_and_resets_timer():
    """asr_reply with speech_start set records ASR latency and resets _speech_start_time."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_latency") as mock_hist,
        patch("inputs.plugins.google_asr.om1_asr_latency_last") as mock_gauge,
        patch("inputs.plugins.google_asr.time.time", return_value=102.0),
    ):
        config = GoogleASRSensorConfig(language="english", api_version="v2")
        sensor = GoogleASRInput(config=config)
        sensor._speech_start_time = 100.0

        sensor._handle_asr_message(json.dumps({"asr_reply": "hello world"}))

        mock_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(2.0)
        mock_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(2.0)
        assert sensor._speech_start_time is None


def test_handle_asr_message_asr_reply_no_latency_without_start_time():
    """asr_reply without prior speech_start does not record latency metrics."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_latency") as mock_hist,
        patch("inputs.plugins.google_asr.om1_asr_latency_last") as mock_gauge,
    ):
        config = GoogleASRSensorConfig()
        sensor = GoogleASRInput(config=config)
        # _speech_start_time is None

        sensor._handle_asr_message(json.dumps({"asr_reply": "hello world"}))

        mock_hist.labels().observe.assert_not_called()
        mock_gauge.labels().set.assert_not_called()


def test_handle_asr_message_full_sequence_records_all_metrics():
    """Full speech_start -> speech_end -> asr_reply sequence records all three metric pairs."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_speech_duration") as mock_dur_hist,
        patch("inputs.plugins.google_asr.om1_asr_speech_duration_last") as mock_dur_gauge,
        patch("inputs.plugins.google_asr.om1_asr_latency") as mock_lat_hist,
        patch("inputs.plugins.google_asr.om1_asr_latency_last") as mock_lat_gauge,
    ):
        config = GoogleASRSensorConfig(language="english", api_version="v2")
        sensor = GoogleASRInput(config=config)
        sensor._speech_start_time = None

        # speech_start — time() == 100.0
        with patch("inputs.plugins.google_asr.time.time", return_value=100.0):
            sensor._handle_asr_message(json.dumps({"type": "speech_start"}))
        assert sensor._speech_start_time == 100.0

        # speech_end — time() == 104.0  =>  duration = 4.0
        with patch("inputs.plugins.google_asr.time.time", return_value=104.0):
            sensor._handle_asr_message(json.dumps({"type": "speech_end"}))
        mock_dur_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(4.0)
        mock_dur_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(4.0)

        # asr_reply — time() == 106.0  =>  latency = 6.0
        with patch("inputs.plugins.google_asr.time.time", return_value=106.0):
            sensor._handle_asr_message(json.dumps({"asr_reply": "hello world"}))
        mock_lat_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(6.0)
        mock_lat_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(6.0)
        assert sensor._speech_start_time is None


def test_handle_asr_message_end_of_utterance_and_asr_reply_sequence():
    """speech_start -> end_of_utterance -> asr_reply sequence records utterance and ASR latency."""
    with (
        patch("inputs.plugins.google_asr.IOProvider"),
        patch("inputs.plugins.google_asr.ASRProvider"),
        patch("inputs.plugins.google_asr.SleepTickerProvider"),
        patch("inputs.plugins.google_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.google_asr.open_zenoh_session"),
        patch("inputs.plugins.google_asr.om1_asr_utterance_end_latency") as mock_utt_hist,
        patch("inputs.plugins.google_asr.om1_asr_utterance_end_latency_last") as mock_utt_gauge,
        patch("inputs.plugins.google_asr.om1_asr_latency") as mock_lat_hist,
        patch("inputs.plugins.google_asr.om1_asr_latency_last") as mock_lat_gauge,
    ):
        config = GoogleASRSensorConfig(language="english", api_version="v2")
        sensor = GoogleASRInput(config=config)
        sensor._speech_start_time = None

        with patch("inputs.plugins.google_asr.time.time", return_value=100.0):
            sensor._handle_asr_message(json.dumps({"type": "speech_start"}))
        with patch("inputs.plugins.google_asr.time.time", return_value=101.5):
            sensor._handle_asr_message(json.dumps({"type": "end_of_utterance"}))
        with patch("inputs.plugins.google_asr.time.time", return_value=103.0):
            sensor._handle_asr_message(json.dumps({"asr_reply": "hello world"}))

        mock_utt_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(1.5)
        mock_utt_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(1.5)
        mock_lat_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(3.0)
        mock_lat_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(3.0)
