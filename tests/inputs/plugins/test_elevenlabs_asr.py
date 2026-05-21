import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.elevenlabs_asr import ElevenLabsASRInput, ElevenLabsASRSensorConfig


def test_initialization_defaults():
    """Test basic initialization with default config."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider") as mock_asr_cls,
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_asr_instance = MagicMock()
        mock_asr_cls.return_value = mock_asr_instance
        mock_session = MagicMock()
        mock_zenoh.return_value = mock_session

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        assert hasattr(sensor, "messages")
        assert isinstance(sensor.messages, list)
        assert sensor.descriptor_for_LLM == "Voice"
        assert sensor._stopped is False
        assert sensor._speech_start_time is None
        mock_asr_instance.start.assert_called_once()
        mock_asr_instance.register_message_callback.assert_called_once_with(sensor._handle_asr_message)


def test_initialization_with_custom_config():
    """Test initialization with custom configuration values."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider") as mock_asr_cls,
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr_cls.return_value = mock_asr_instance

        config = ElevenLabsASRSensorConfig(
            api_key="test_key",
            rate=16000,
            chunk=8192,
            base_url="wss://custom.example.com/asr",
            microphone_device_id=2,
            microphone_name="my_mic",
            language="english",
            remote_input=True,
            enable_tts_interrupt=True,
        )
        ElevenLabsASRInput(config=config)

        mock_asr_cls.assert_called_once_with(
            rate=16000,
            chunk=8192,
            ws_url="wss://custom.example.com/asr",
            device_id=2,
            microphone_name="my_mic",
            language_code="en",
            remote_input=True,
            enable_tts_interrupt=True,
        )


def test_initialization_default_base_url_uses_api_key():
    """Test that default base_url includes the api_key."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider") as mock_asr_cls,
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr_cls.return_value = mock_asr_instance

        config = ElevenLabsASRSensorConfig(api_key="my_secret_key")
        ElevenLabsASRInput(config=config)

        call_kwargs = mock_asr_cls.call_args.kwargs
        assert "my_secret_key" in call_kwargs["ws_url"]


def test_initialization_unsupported_language_falls_back_to_auto():
    """Test that unsupported language defaults to auto."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider") as mock_asr_cls,
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr_cls.return_value = mock_asr_instance

        config = ElevenLabsASRSensorConfig(language="klingon")
        ElevenLabsASRInput(config=config)

        call_kwargs = mock_asr_cls.call_args.kwargs
        assert call_kwargs["language_code"] == "auto"


def test_initialization_zenoh_failure():
    """Test initialization when Zenoh fails – sensor is still usable."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_zenoh.side_effect = Exception("Zenoh not available")

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        assert sensor.session is None
        assert sensor.asr_publisher is None


def test_initialization_zenoh_publisher_declared():
    """Test that Zenoh publisher is declared on the correct topic."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        mock_session.declare_publisher.assert_called_once_with("om/asr/text")
        assert sensor.asr_publisher is mock_publisher


# ---------------------------------------------------------------------------
# _handle_asr_message tests
# ---------------------------------------------------------------------------


def test_handle_asr_message_committed_multi_word():
    """committed message with >1 word is queued."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "asr_reply": "hello world test"})
        sensor._handle_asr_message(raw)

        assert not sensor.message_buffer.empty()
        assert sensor.message_buffer.get_nowait() == "hello world test"


def test_handle_asr_message_committed_single_word_ignored():
    """committed message with a single word is NOT queued."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "asr_reply": "hello"})
        sensor._handle_asr_message(raw)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_partial_sets_speech_start_time():
    """partial message sets _speech_start_time."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        before = time.time()
        raw = json.dumps({"type": "partial", "asr_reply": ""})
        sensor._handle_asr_message(raw)
        after = time.time()

        assert sensor._speech_start_time is not None
        assert before <= sensor._speech_start_time <= after


def test_handle_asr_message_committed_records_latency():
    """committed message records latency and resets _speech_start_time."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
        patch("inputs.plugins.elevenlabs_asr.om1_asr_latency") as mock_latency,
        patch("inputs.plugins.elevenlabs_asr.om1_asr_latency_last") as mock_latency_last,
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor._speech_start_time = time.time() - 0.5  # 500 ms ago

        raw = json.dumps({"type": "committed", "asr_reply": "hello world"})
        sensor._handle_asr_message(raw)

        mock_latency.labels.assert_called_once()
        mock_latency_last.labels.assert_called_once()
        assert sensor._speech_start_time is None


def test_handle_asr_message_stopped_ignores_message():
    """When _stopped is True, messages are not queued."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor._stopped = True

        raw = json.dumps({"type": "committed", "asr_reply": "hello world test"})
        sensor._handle_asr_message(raw)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_invalid_json():
    """Invalid JSON is silently ignored."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        sensor._handle_asr_message("not valid json!!!")

        assert sensor.message_buffer.empty()


def test_handle_asr_message_no_asr_reply_field():
    """Message without asr_reply field is ignored."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "other_key": "other_value"})
        sensor._handle_asr_message(raw)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_committed_cjk_chinese_accepted():
    """committed message with Chinese text longer than 2 chars is accepted."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "asr_reply": "你好吗"})
        sensor._handle_asr_message(raw)

        assert not sensor.message_buffer.empty()
        assert sensor.message_buffer.get_nowait() == "你好吗"


def test_handle_asr_message_committed_cjk_chinese_too_short_rejected():
    """committed message with 2 or fewer Chinese characters is rejected."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "asr_reply": "你好"})
        sensor._handle_asr_message(raw)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_committed_cjk_japanese_accepted():
    """committed message with Japanese hiragana text longer than 2 chars is accepted."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        # Japanese hiragana: "こんにちは" (5 chars) -> accepted
        raw = json.dumps({"type": "committed", "asr_reply": "こんにちは"})
        sensor._handle_asr_message(raw)

        assert not sensor.message_buffer.empty()
        assert sensor.message_buffer.get_nowait() == "こんにちは"


def test_handle_asr_message_committed_cjk_korean_accepted():
    """committed message with Korean hangul text longer than 2 chars is accepted."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "asr_reply": "안녕하세요"})
        sensor._handle_asr_message(raw)

        assert not sensor.message_buffer.empty()
        assert sensor.message_buffer.get_nowait() == "안녕하세요"


def test_handle_asr_message_committed_cjk_single_char_rejected():
    """committed message with a single CJK character is rejected."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "asr_reply": "你"})
        sensor._handle_asr_message(raw)

        assert sensor.message_buffer.empty()


def test_handle_asr_message_committed_cjk_mixed_with_latin_accepted():
    """committed message with mixed CJK and Latin text longer than 2 chars is accepted via CJK path."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        raw = json.dumps({"type": "committed", "asr_reply": "hello你好"})
        sensor._handle_asr_message(raw)

        assert not sensor.message_buffer.empty()
        assert sensor.message_buffer.get_nowait() == "hello你好"


def test_handle_asr_message_no_speech_start_time_skips_latency():
    """committed message with no prior partial does not call latency metrics."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
        patch("inputs.plugins.elevenlabs_asr.om1_asr_latency") as mock_latency,
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        assert sensor._speech_start_time is None

        raw = json.dumps({"type": "committed", "asr_reply": "hello world"})
        sensor._handle_asr_message(raw)

        mock_latency.labels.assert_not_called()


@pytest.mark.asyncio
async def test_poll_returns_message_when_available():
    """_poll returns the next message from the queue."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        await sensor.message_buffer.put("hi there")
        result = await sensor._poll()
        assert result == "hi there"


@pytest.mark.asyncio
async def test_poll_returns_none_when_empty():
    """_poll returns None when the queue is empty."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        with patch("inputs.plugins.elevenlabs_asr.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_poll_sleeps_when_empty():
    """_poll calls asyncio.sleep(0.01) when the queue is empty."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        with patch("inputs.plugins.elevenlabs_asr.asyncio.sleep", new=AsyncMock()) as mock_sleep:
            await sensor._poll()
            mock_sleep.assert_called_once_with(0.01)


@pytest.mark.asyncio
async def test_raw_to_text_none_returns_none():
    """_raw_to_text(None) returns None."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_valid_input():
    """_raw_to_text returns a Message with correct content and timestamp."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
        patch("inputs.plugins.elevenlabs_asr.time.time", return_value=999.0),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        result = await sensor._raw_to_text("hello world")
        assert result is not None
        assert isinstance(result, Message)
        assert result.message == "hello world"
        assert result.timestamp == 999.0


@pytest.mark.asyncio
async def test_raw_to_text_wrapper_first_message():
    """raw_to_text appends first message to empty buffer."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        await sensor.raw_to_text("first message")

        assert len(sensor.messages) == 1
        assert sensor.messages[0] == "first message"


@pytest.mark.asyncio
async def test_raw_to_text_wrapper_appends_to_existing():
    """raw_to_text appends subsequent message to existing entry."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor.messages.append("first part")

        await sensor.raw_to_text("second part")

        assert len(sensor.messages) == 1
        assert sensor.messages[0] == "first part second part"


@pytest.mark.asyncio
async def test_raw_to_text_none_sets_skip_sleep_when_messages_exist():
    """raw_to_text(None) sets skip_sleep=True when messages buffer is non-empty."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider") as mock_sleep_ticker_cls,
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        mock_sleep_ticker = MagicMock()
        mock_sleep_ticker_cls.return_value = mock_sleep_ticker

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor.messages.append("some existing message")

        await sensor.raw_to_text(None)

        assert mock_sleep_ticker.skip_sleep is True


@pytest.mark.asyncio
async def test_raw_to_text_none_does_not_set_skip_sleep_when_buffer_empty():
    """raw_to_text(None) does not touch skip_sleep when messages buffer is empty."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider") as mock_sleep_ticker_cls,
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        mock_sleep_ticker = MagicMock()
        mock_sleep_ticker.skip_sleep = False
        mock_sleep_ticker_cls.return_value = mock_sleep_ticker

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        await sensor.raw_to_text(None)

        assert mock_sleep_ticker.skip_sleep is False


def test_formatted_latest_buffer_empty():
    """Returns None when messages buffer is empty."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        assert sensor.formatted_latest_buffer() is None


def test_formatted_latest_buffer_formats_and_clears():
    """Returns formatted string and clears the buffer."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider") as mock_io_cls,
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider") as mock_conv_cls,
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        mock_io = MagicMock()
        mock_io_cls.return_value = mock_io
        mock_conv = MagicMock()
        mock_conv_cls.return_value = mock_conv

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor.messages.append("hello world how are you")

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Voice" in result
        assert "hello world how are you" in result
        assert len(sensor.messages) == 0

        mock_io.add_input.assert_called_once_with("Voice", "hello world how are you", pytest.approx(time.time(), abs=2))
        mock_io.add_mode_transition_input.assert_called_once_with("hello world how are you")
        mock_conv.store_user_message.assert_called_once_with("hello world how are you")


def test_formatted_latest_buffer_publishes_to_zenoh():
    """When Zenoh is initialised, formatted_latest_buffer publishes the message."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
        patch("inputs.plugins.elevenlabs_asr.ASRText") as mock_asr_text_cls,
        patch("inputs.plugins.elevenlabs_asr.prepare_header") as mock_header,
    ):
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        mock_asr_msg = MagicMock()
        mock_asr_msg.serialize.return_value = b"serialized_payload"
        mock_asr_text_cls.return_value = mock_asr_msg
        mock_header.return_value = {"id": "uuid-test"}

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor.messages.append("publish this message")

        result = sensor.formatted_latest_buffer()

        assert result is not None
        mock_publisher.put.assert_called_once_with(b"serialized_payload")


def test_formatted_latest_buffer_zenoh_publish_failure_does_not_raise():
    """Zenoh publish failure is logged but does not propagate."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.put.side_effect = Exception("publish error")
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor.messages.append("some message")

        result = sensor.formatted_latest_buffer()
        assert result is not None  # still returns the formatted string


def test_stop_sets_stopped_and_clears_buffers():
    """stop() sets _stopped, drains queue, and clears messages."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider") as mock_asr_cls,
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_asr_instance = MagicMock()
        mock_asr_cls.return_value = mock_asr_instance
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        sensor.message_buffer.put_nowait("msg1")
        sensor.message_buffer.put_nowait("msg2")
        sensor.messages.append("buffered")

        sensor.stop()

        assert sensor._stopped is True
        assert sensor.message_buffer.empty()
        assert len(sensor.messages) == 0
        mock_asr_instance.unregister_message_callback.assert_called_once()
        mock_publisher.undeclare.assert_called_once()
        mock_session.close.assert_called_once()


def test_stop_without_zenoh():
    """stop() succeeds gracefully when Zenoh was never initialized."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_zenoh.side_effect = Exception("Zenoh not available")

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True


def test_stop_asr_unregister_failure_does_not_raise():
    """stop() handles ASR unregister failure gracefully."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider") as mock_asr_cls,
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_asr_instance = MagicMock()
        mock_asr_instance.unregister_message_callback.side_effect = Exception("unregister error")
        mock_asr_cls.return_value = mock_asr_instance
        mock_session = MagicMock()
        mock_session.declare_publisher.return_value = MagicMock()
        mock_zenoh.return_value = mock_session

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True


def test_stop_zenoh_undeclare_failure_does_not_raise():
    """stop() handles Zenoh undeclare failure gracefully."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.undeclare.side_effect = Exception("undeclare error")
        mock_session.declare_publisher.return_value = mock_publisher
        mock_zenoh.return_value = mock_session

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True
        mock_session.close.assert_called_once()


def test_stop_zenoh_close_failure_does_not_raise():
    """stop() handles Zenoh session close failure gracefully."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = MagicMock()
        mock_session.close.side_effect = Exception("close error")
        mock_session.declare_publisher.return_value = MagicMock()
        mock_zenoh.return_value = mock_session

        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)

        sensor.stop()
        assert sensor._stopped is True


def test_stopped_sensor_ignores_further_messages():
    """After stop(), _handle_asr_message silently drops new messages."""
    with (
        patch("inputs.plugins.elevenlabs_asr.IOProvider"),
        patch("inputs.plugins.elevenlabs_asr.ASRProvider"),
        patch("inputs.plugins.elevenlabs_asr.SleepTickerProvider"),
        patch("inputs.plugins.elevenlabs_asr.TeleopsConversationProvider"),
        patch("inputs.plugins.elevenlabs_asr.open_zenoh_session"),
    ):
        config = ElevenLabsASRSensorConfig()
        sensor = ElevenLabsASRInput(config=config)
        sensor.stop()

        sensor._handle_asr_message(json.dumps({"type": "committed", "asr_reply": "hello world"}))

        assert sensor.message_buffer.empty()
