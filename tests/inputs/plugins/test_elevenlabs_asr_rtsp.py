import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.elevenlabs_asr_rtsp import ElevenLabsASRRTSPInput, ElevenLabsASRRTSPSensorConfig


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_asr_provider():
    mock_constructor = MagicMock()
    mock_instance = MagicMock()
    mock_constructor.return_value = mock_instance
    return mock_constructor, mock_instance


@pytest.fixture
def mock_sleep_ticker_provider():
    mock_constructor = MagicMock()
    mock_instance = MagicMock()
    mock_constructor.return_value = mock_instance
    return mock_constructor, mock_instance


@pytest.fixture
def mock_teleops_conversation_provider():
    mock_constructor = MagicMock()
    mock_instance = MagicMock()
    mock_constructor.return_value = mock_instance
    return mock_constructor, mock_instance


@pytest.fixture
def mock_zenoh():
    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session") as mock_open_session,
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRText") as mock_asr_text,
        patch("inputs.plugins.elevenlabs_asr_rtsp.prepare_header") as mock_prepare_header,
    ):
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_open_session.return_value = mock_session
        mock_session.declare_publisher.return_value = mock_publisher

        yield {
            "open_session": mock_open_session,
            "session": mock_session,
            "publisher": mock_publisher,
            "asr_text_cls": mock_asr_text,
            "prepare_header": mock_prepare_header,
        }


def _build_sensor(
    config, mock_io_provider, mock_asr_provider, mock_sleep_ticker_provider, mock_teleops_conv_provider, mock_zenoh
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conv_provider
    mock_asr_constructor, _ = mock_asr_provider
    mock_sleep_constructor, _ = mock_sleep_ticker_provider
    mock_conv_constructor, _ = mock_teleops_conv_provider

    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", return_value=mock_asr_instance),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
    ):
        return ElevenLabsASRRTSPInput(config=config)


def test_initialization_creates_providers_and_buffers(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """Initialization wires up all providers and Zenoh publisher."""
    mock_asr_constructor, mock_asr_instance = mock_asr_provider
    mock_sleep_constructor, _ = mock_sleep_ticker_provider
    mock_conv_constructor, _ = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig()

    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", new=mock_sleep_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", new=mock_conv_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
    ):
        sensor = ElevenLabsASRRTSPInput(config=config)

    assert sensor.descriptor_for_LLM == "Voice"
    assert isinstance(sensor.messages, list)
    assert hasattr(sensor, "message_buffer")
    assert sensor._stopped is False
    assert sensor._speech_start_time is None
    assert sensor.session is mock_zenoh["session"]
    assert sensor.asr_publisher is mock_zenoh["publisher"]
    mock_asr_instance.start.assert_called_once()
    mock_asr_instance.register_message_callback.assert_called_once_with(sensor._handle_asr_message)
    mock_zenoh["session"].declare_publisher.assert_called_once_with("om/asr/text")


def test_initialization_default_base_url_uses_api_key(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """When base_url is not set, the default URL embeds the api_key."""
    mock_asr_constructor, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig(api_key="my_secret_key")

    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
    ):
        ElevenLabsASRRTSPInput(config=config)

    call_kwargs = mock_asr_constructor.call_args.kwargs
    assert "my_secret_key" in call_kwargs["ws_url"]


def test_initialization_unsupported_language_falls_back_to_auto(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """Unsupported language value defaults language_code to 'auto'."""
    mock_asr_constructor, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig(language="klingon")

    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
    ):
        ElevenLabsASRRTSPInput(config=config)

    call_kwargs = mock_asr_constructor.call_args.kwargs
    assert call_kwargs["language_code"] == "auto"


def test_initialization_zenoh_failure(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
):
    """When Zenoh fails to initialize, session and publisher are None."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig()

    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", return_value=mock_asr_instance),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", side_effect=Exception("Zenoh down")),
    ):
        sensor = ElevenLabsASRRTSPInput(config=config)

    assert sensor.session is None
    assert sensor.asr_publisher is None


def test_handle_asr_message_committed_multi_word(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """committed message with >1 word is queued."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    raw = json.dumps({"type": "committed", "asr_reply": "hello world test"})
    sensor._handle_asr_message(raw)

    assert not sensor.message_buffer.empty()
    assert sensor.message_buffer.get_nowait() == "hello world test"


def test_handle_asr_message_committed_single_word_ignored(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """committed message with a single word is NOT queued."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    raw = json.dumps({"type": "committed", "asr_reply": "hello"})
    sensor._handle_asr_message(raw)

    assert sensor.message_buffer.empty()


def test_handle_asr_message_partial_sets_speech_start_time(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """partial message records speech start time."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    before = time.time()
    raw = json.dumps({"type": "partial", "asr_reply": ""})
    sensor._handle_asr_message(raw)
    after = time.time()

    assert sensor._speech_start_time is not None
    assert before <= sensor._speech_start_time <= after


def test_handle_asr_message_committed_records_latency_and_resets(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """committed message records latency metrics and resets _speech_start_time."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.om1_asr_latency") as mock_latency,
        patch("inputs.plugins.elevenlabs_asr_rtsp.om1_asr_latency_last") as mock_latency_last,
    ):
        sensor._speech_start_time = time.time() - 0.5

        raw = json.dumps({"type": "committed", "asr_reply": "hello world"})
        sensor._handle_asr_message(raw)

        mock_latency.labels.assert_called_once()
        mock_latency_last.labels.assert_called_once()
        assert sensor._speech_start_time is None


def test_handle_asr_message_no_speech_start_time_skips_latency(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """committed without a prior partial does not call latency metrics."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    with patch("inputs.plugins.elevenlabs_asr_rtsp.om1_asr_latency") as mock_latency:
        assert sensor._speech_start_time is None
        raw = json.dumps({"type": "committed", "asr_reply": "hello world"})
        sensor._handle_asr_message(raw)
        mock_latency.labels.assert_not_called()


def test_handle_asr_message_stopped_ignores_message(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """When _stopped is True, messages are silently dropped."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )
    sensor._stopped = True

    raw = json.dumps({"type": "committed", "asr_reply": "hello world test"})
    sensor._handle_asr_message(raw)

    assert sensor.message_buffer.empty()


def test_handle_asr_message_invalid_json(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """Invalid JSON is silently ignored."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    sensor._handle_asr_message("not valid json!!!")
    assert sensor.message_buffer.empty()


def test_handle_asr_message_no_asr_reply_field(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """committed message without asr_reply field is ignored."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    raw = json.dumps({"type": "committed", "other_key": "value"})
    sensor._handle_asr_message(raw)

    assert sensor.message_buffer.empty()


@pytest.mark.asyncio
async def test_poll_returns_message_when_available(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """_poll returns the next message from the queue."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    sensor.message_buffer.put_nowait("hi there")
    result = await sensor._poll()
    assert result == "hi there"


@pytest.mark.asyncio
async def test_poll_returns_none_when_empty(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """_poll returns None when the queue is empty."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    result = await sensor._poll()
    assert result is None


@pytest.mark.asyncio
async def test_poll_sleeps_when_empty(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """_poll calls asyncio.sleep(0.01) when the queue is empty."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    with patch("inputs.plugins.elevenlabs_asr_rtsp.asyncio.sleep", new=AsyncMock()) as mock_sleep:
        await sensor._poll()
        mock_sleep.assert_called_once_with(0.01)


@pytest.mark.asyncio
async def test_poll_returns_none_when_stopped(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """_poll returns None immediately when sensor is stopped."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )
    sensor._stopped = True
    sensor.message_buffer.put_nowait("should be ignored")

    result = await sensor._poll()
    assert result is None


# ---------------------------------------------------------------------------
# _raw_to_text / raw_to_text tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_to_text_none_returns_none(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """_raw_to_text(None) returns None."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    result = await sensor._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_valid_string(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """_raw_to_text returns a Message with correct content and timestamp."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    before = time.time()
    result = await sensor._raw_to_text("hello world")
    after = time.time()

    assert result is not None
    assert isinstance(result, Message)
    assert result.message == "hello world"
    assert before <= result.timestamp <= after


@pytest.mark.asyncio
async def test_raw_to_text_wrapper_first_message(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """raw_to_text appends first message to empty buffer."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    await sensor.raw_to_text("first message")

    assert len(sensor.messages) == 1
    assert sensor.messages[0] == "first message"


@pytest.mark.asyncio
async def test_raw_to_text_wrapper_appends_to_existing(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """raw_to_text concatenates subsequent message to existing buffer entry."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )
    sensor.messages.append("first part")

    await sensor.raw_to_text("second part")

    assert len(sensor.messages) == 1
    assert sensor.messages[0] == "first part second part"


@pytest.mark.asyncio
async def test_raw_to_text_none_sets_skip_sleep_when_messages_exist(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """raw_to_text(None) sets skip_sleep=True when buffer is non-empty."""
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_asr_instance = mock_asr_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", return_value=mock_asr_instance),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
    ):
        sensor = ElevenLabsASRRTSPInput(config=config)

    sensor.messages.append("existing message")
    mock_sleep_ticker_instance.skip_sleep = False

    await sensor.raw_to_text(None)

    assert mock_sleep_ticker_instance.skip_sleep is True


@pytest.mark.asyncio
async def test_raw_to_text_none_does_not_set_skip_sleep_when_buffer_empty(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """raw_to_text(None) does not touch skip_sleep when buffer is empty."""
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_asr_instance = mock_asr_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", return_value=mock_asr_instance),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
    ):
        sensor = ElevenLabsASRRTSPInput(config=config)

    mock_sleep_ticker_instance.skip_sleep = False

    await sensor.raw_to_text(None)

    assert mock_sleep_ticker_instance.skip_sleep is False


def test_formatted_latest_buffer_empty(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """Returns None when messages buffer is empty."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    assert sensor.formatted_latest_buffer() is None


def test_formatted_latest_buffer_formats_and_clears(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """Returns formatted string, calls IO/conversation providers, and clears buffer."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig()
    fixed_timestamp = 1234.0

    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", return_value=mock_asr_instance),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
        patch("time.time", return_value=fixed_timestamp),
    ):
        sensor = ElevenLabsASRRTSPInput(config=config)

        msg_content = "final transcription result"
        sensor.messages = [msg_content]
        result = sensor.formatted_latest_buffer()

    assert result is not None
    assert "Voice" in result
    assert msg_content in result
    assert len(sensor.messages) == 0

    mock_io_provider.add_input.assert_called_once_with("Voice", msg_content, fixed_timestamp)
    mock_io_provider.add_mode_transition_input.assert_called_once_with(msg_content)
    mock_teleops_conv_instance.store_user_message.assert_called_once_with(msg_content)


def test_formatted_latest_buffer_publishes_to_zenoh(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """When Zenoh is available, formatted_latest_buffer publishes the message."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    mock_asr_msg = MagicMock()
    mock_asr_msg.serialize.return_value = b"serialized_payload"
    mock_zenoh["asr_text_cls"].return_value = mock_asr_msg

    sensor.messages.append("publish this")
    result = sensor.formatted_latest_buffer()

    assert result is not None
    mock_zenoh["publisher"].put.assert_called_once_with(b"serialized_payload")


def test_formatted_latest_buffer_zenoh_publish_failure_does_not_raise(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """Zenoh publish errors are logged but do not propagate."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )
    mock_zenoh["publisher"].put.side_effect = Exception("publish error")

    sensor.messages.append("some message")
    result = sensor.formatted_latest_buffer()

    assert result is not None


def test_stop_sets_stopped_and_clears_buffers(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """stop() sets _stopped, drains queue, clears messages, and tears down resources."""
    mock_asr_constructor, mock_asr_instance = mock_asr_provider
    mock_sleep_constructor, _ = mock_sleep_ticker_provider
    mock_conv_constructor, _ = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", new=mock_sleep_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", new=mock_conv_constructor),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", mock_zenoh["open_session"]),
    ):
        sensor = ElevenLabsASRRTSPInput(config=config)

    sensor.message_buffer.put_nowait("msg1")
    sensor.message_buffer.put_nowait("msg2")
    sensor.messages.append("buffered")

    sensor.stop()

    assert sensor._stopped is True
    assert sensor.message_buffer.empty()
    assert len(sensor.messages) == 0
    mock_asr_instance.unregister_message_callback.assert_called_once()
    mock_zenoh["publisher"].undeclare.assert_called_once()
    mock_zenoh["session"].close.assert_called_once()


def test_stop_without_zenoh(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
):
    """stop() succeeds gracefully when Zenoh was never initialized."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = ElevenLabsASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.elevenlabs_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.elevenlabs_asr_rtsp.ASRRTSPProvider", return_value=mock_asr_instance),
        patch("inputs.plugins.elevenlabs_asr_rtsp.SleepTickerProvider", return_value=mock_sleep_ticker_instance),
        patch(
            "inputs.plugins.elevenlabs_asr_rtsp.TeleopsConversationProvider", return_value=mock_teleops_conv_instance
        ),
        patch("inputs.plugins.elevenlabs_asr_rtsp.open_zenoh_session", side_effect=Exception("Zenoh down")),
    ):
        sensor = ElevenLabsASRRTSPInput(config=config)

    sensor.stop()
    assert sensor._stopped is True


def test_stop_asr_unregister_failure_does_not_raise(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """stop() handles ASR unregister failure gracefully."""
    _, mock_asr_instance = mock_asr_provider
    mock_asr_instance.unregister_message_callback.side_effect = Exception("unregister error")

    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    sensor.stop()
    assert sensor._stopped is True


def test_stop_zenoh_undeclare_failure_does_not_raise(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """stop() handles Zenoh undeclare failure gracefully and still closes session."""
    mock_zenoh["publisher"].undeclare.side_effect = Exception("undeclare error")

    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    sensor.stop()
    assert sensor._stopped is True
    mock_zenoh["session"].close.assert_called_once()


def test_stop_zenoh_close_failure_does_not_raise(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """stop() handles Zenoh session close failure gracefully."""
    mock_zenoh["session"].close.side_effect = Exception("close error")

    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    sensor.stop()
    assert sensor._stopped is True


def test_stop_all_failures_does_not_raise(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """stop() handles all resource teardown failures gracefully."""
    _, mock_asr_instance = mock_asr_provider
    mock_asr_instance.unregister_message_callback.side_effect = Exception("unregister error")
    mock_zenoh["publisher"].undeclare.side_effect = Exception("undeclare error")
    mock_zenoh["session"].close.side_effect = Exception("close error")

    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )

    sensor.stop()
    assert sensor._stopped is True


def test_stopped_sensor_ignores_further_messages(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """After stop(), _handle_asr_message silently drops new messages."""
    sensor = _build_sensor(
        ElevenLabsASRRTSPSensorConfig(),
        mock_io_provider,
        mock_asr_provider,
        mock_sleep_ticker_provider,
        mock_teleops_conversation_provider,
        mock_zenoh,
    )
    sensor.stop()

    sensor._handle_asr_message(json.dumps({"type": "committed", "asr_reply": "hello world"}))

    assert sensor.message_buffer.empty()
