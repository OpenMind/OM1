import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.riva_asr import RivaASRInput, RivaASRSensorConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        assert hasattr(sensor, "messages")
        assert sensor.descriptor_for_LLM == "Voice"
        assert sensor._stopped is False
        mock_asr_instance.start.assert_called_once()
        mock_asr_instance.register_message_callback.assert_called_once()


def test_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = RivaASRSensorConfig(
            api_key="test_key",
            rate=16000,
            chunk=8192,
            base_url="wss://test.com",
            microphone_device_id=1,
            microphone_name="test_mic",
            remote_input=True,
            enable_tts_interrupt=True,
        )
        sensor = RivaASRInput(config=config)

        assert hasattr(sensor, "messages")
        mock_asr.assert_called_with(
            rate=16000,
            chunk=8192,
            ws_url="wss://test.com",
            device_id=1,
            microphone_name="test_mic",
            remote_input=True,
            enable_tts_interrupt=True,
        )


def test_initialization_zenoh_failure():
    """Test initialization when Zenoh fails to initialize."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance
        mock_zenoh.side_effect = Exception("Zenoh connection failed")

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        assert sensor.session is None
        assert sensor.asr_publisher is None


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        with patch("inputs.plugins.riva_asr.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()
            assert result is None


@pytest.mark.asyncio
async def test_poll_with_message():
    """Test _poll method when message is available."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        await sensor.message_buffer.put("test message")

        result = await sensor._poll()
        assert result == "test message"


def test_handle_asr_message_valid():
    """Test _handle_asr_message with valid message."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        raw_message = json.dumps({"asr_reply": "hello world test"})
        sensor._handle_asr_message(raw_message)

        assert not sensor.message_buffer.empty()
        result = sensor.message_buffer.get_nowait()
        assert result == "hello world test"


def test_handle_asr_message_single_word():
    """Test _handle_asr_message with single word (should be ignored)."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        raw_message = json.dumps({"asr_reply": "hello"})
        sensor._handle_asr_message(raw_message)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_stopped():
    """Test _handle_asr_message when sensor is stopped."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)
        sensor._stopped = True

        raw_message = json.dumps({"asr_reply": "hello world test"})
        sensor._handle_asr_message(raw_message)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_invalid_json():
    """Test _handle_asr_message with invalid JSON."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        sensor._handle_asr_message("not a valid json")

        assert sensor.message_buffer.empty()


def test_handle_asr_message_no_asr_reply():
    """Test _handle_asr_message with missing asr_reply field."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        raw_message = json.dumps({"other_field": "value"})
        sensor._handle_asr_message(raw_message)

        assert sensor.message_buffer.empty()


@pytest.mark.asyncio
async def test_raw_to_text_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_valid():
    """Test _raw_to_text with valid input."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
        patch("inputs.plugins.riva_asr.time.time", return_value=123.456),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        result = await sensor._raw_to_text("test message")
        assert result is not None
        assert result.message == "test message"
        assert result.timestamp == 123.456


@pytest.mark.asyncio
async def test_raw_to_text_wrapper_none():
    """Test raw_to_text with None input."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider") as mock_sleep_ticker,
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        mock_sleep_ticker_instance = MagicMock()
        mock_sleep_ticker.return_value = mock_sleep_ticker_instance

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)
        sensor.messages.append("existing message")

        await sensor.raw_to_text(None)

        assert mock_sleep_ticker_instance.skip_sleep is True


@pytest.mark.asyncio
async def test_raw_to_text_wrapper_first_message():
    """Test raw_to_text with first message."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        await sensor.raw_to_text("first message")

        assert len(sensor.messages) == 1
        assert sensor.messages[0] == "first message"


@pytest.mark.asyncio
async def test_raw_to_text_wrapper_append_message():
    """Test raw_to_text appending to existing message."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)
        sensor.messages.append("first message")

        await sensor.raw_to_text("second message")

        assert len(sensor.messages) == 1
        assert sensor.messages[0] == "first message second message"


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider") as mock_io,
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider") as mock_conv,
        patch("inputs.plugins.riva_asr.open_zenoh_session"),
    ):

        mock_io_instance = MagicMock()
        mock_io.return_value = mock_io_instance
        mock_conv_instance = MagicMock()
        mock_conv.return_value = mock_conv_instance

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None

        test_message = Message(timestamp=123.456, message="hello world how are you")
        sensor.messages = []  # type: ignore
        sensor.messages.append(test_message.message)  # type: ignore

        result = sensor.formatted_latest_buffer()
        assert isinstance(result, str)
        assert "INPUT:" in result
        assert "Voice" in result
        assert "hello world how are you" in result
        assert "// START" in result
        assert "// END" in result
        assert len(sensor.messages) == 0

        mock_io_instance.add_input.assert_called_once()
        mock_io_instance.add_mode_transition_input.assert_called_once()
        mock_conv_instance.store_user_message.assert_called_once()


def test_formatted_latest_buffer_with_zenoh():
    """Test formatted_latest_buffer with Zenoh publishing."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
        patch("inputs.plugins.riva_asr.ASRText") as mock_asr_text,
        patch("inputs.plugins.riva_asr.prepare_header") as mock_header,
    ):

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        mock_asr_msg = MagicMock()
        mock_asr_msg.serialize.return_value = b"serialized"
        mock_asr_text.return_value = mock_asr_msg
        mock_header.return_value = {"id": "test"}

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        sensor.messages.append("test message")
        result = sensor.formatted_latest_buffer()

        assert result is not None
        mock_publisher.put.assert_called_once_with(b"serialized")


def test_formatted_latest_buffer_zenoh_failure():
    """Test formatted_latest_buffer when Zenoh publishing fails."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.put.side_effect = Exception("Zenoh publish failed")
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        sensor.messages.append("test message")
        result = sensor.formatted_latest_buffer()
        assert result is not None


def test_stop():
    """Test stop method."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

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


def test_stop_asr_unregister_failure():
    """Test stop method when ASR unregister fails."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_asr_instance = MagicMock()
        mock_asr_instance.unregister_message_callback.side_effect = Exception(
            "Unregister failed"
        )
        mock_asr.return_value = mock_asr_instance

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True


def test_stop_asr_stop_failure():
    """Test stop method when ASR stop fails."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_asr_instance = MagicMock()
        mock_asr_instance.stop.side_effect = Exception("Stop failed")
        mock_asr.return_value = mock_asr_instance

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True


def test_stop_zenoh_undeclare_failure():
    """Test stop method when Zenoh undeclare fails."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.undeclare.side_effect = Exception("Undeclare failed")
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True
        mock_session.close.assert_called_once()


def test_stop_without_zenoh():
    """Test stop method when Zenoh was never initialized."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
        patch("inputs.plugins.riva_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.riva_asr.open_zenoh_session") as mock_zenoh,
    ):

        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance
        mock_zenoh.side_effect = Exception("Zenoh init failed")

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True
