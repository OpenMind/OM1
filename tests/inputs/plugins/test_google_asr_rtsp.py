import time
from unittest.mock import Mock, patch

import pytest

from inputs.plugins.google_asr_rtsp import GoogleASRRTSPInput, GoogleASRRTSPSensorConfig


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.google_asr_rtsp.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_asr_provider():
    mock_constructor = Mock()
    mock_instance = Mock()
    mock_constructor.return_value = mock_instance
    return mock_constructor, mock_instance


@pytest.fixture
def mock_sleep_ticker_provider():
    mock_constructor = Mock()
    mock_instance = Mock()
    mock_constructor.return_value = mock_instance
    return mock_constructor, mock_instance


@pytest.fixture
def mock_teleops_conversation_provider():
    mock_constructor = Mock()
    mock_instance = Mock()
    mock_constructor.return_value = mock_instance
    return mock_constructor, mock_instance


@pytest.fixture
def mock_zenoh():
    with (
        patch("inputs.plugins.google_asr_rtsp.open_zenoh_session") as mock_open_session,
        patch("inputs.plugins.google_asr_rtsp.ASRText") as mock_asr_text,
        patch("inputs.plugins.google_asr_rtsp.prepare_header") as mock_prepare_header,
    ):
        mock_session_instance = Mock()
        mock_publisher_instance = Mock()
        mock_open_session.return_value = mock_session_instance
        mock_session_instance.declare_publisher.return_value = mock_publisher_instance

        yield {
            "open_session": mock_open_session,
            "session": mock_session_instance,
            "publisher": mock_publisher_instance,
            "asr_text_cls": mock_asr_text,
            "prepare_header": mock_prepare_header,
        }


def test_initialization_creates_providers_and_buffers(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, mock_asr_instance = mock_asr_provider
    mock_sleep_ticker_constructor, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    mock_teleops_conv_constructor, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    api_key = config.api_key
    rtsp_url = config.rtsp_url
    rate = config.rate
    chunk = config.chunk
    enable_tts_interrupt = config.enable_tts_interrupt

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            new=mock_sleep_ticker_constructor,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            new=mock_teleops_conv_constructor,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once_with(
        rtsp_url=rtsp_url,
        rate=rate,
        chunk=chunk,
        ws_url=f"wss://api.openmind.com/api/core/google/asr/v2?api_key={api_key}",
        language_code="en-US",
        alternative_language_codes=[],
        enable_tts_interrupt=enable_tts_interrupt,
    )
    mock_asr_instance.start.assert_called_once()
    mock_asr_instance.register_message_callback.assert_called_once()

    mock_sleep_ticker_constructor.assert_called_once()
    mock_teleops_conv_constructor.assert_called_once_with(api_key=api_key)

    mock_zenoh["open_session"].assert_called_once()
    mock_zenoh["session"].declare_publisher.assert_called_once_with("om/asr/text")

    assert instance.io_provider is not None
    assert mock_io_provider is not None
    assert isinstance(instance.messages, list)
    assert hasattr(instance, "message_buffer")
    assert instance.descriptor_for_LLM == "Voice"
    assert instance.session is mock_zenoh["session"]
    assert instance.asr_publisher is mock_zenoh["publisher"]


@pytest.mark.asyncio
async def test_poll_returns_message_from_buffer(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    test_message = "Hello world"
    instance.message_buffer.put_nowait(test_message)

    result = await instance._poll()

    assert result == test_message


@pytest.mark.asyncio
async def test_poll_returns_none_if_buffer_empty(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    result = await instance._poll()

    assert result is None


@pytest.mark.asyncio
async def test_poll_has_delay(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    with patch("inputs.plugins.google_asr_rtsp.asyncio.sleep") as mock_sleep:
        await instance._poll()
        mock_sleep.assert_called_once_with(0.01)


def test_handle_asr_message_processes_valid_json_with_asr_reply_longer_than_one_word(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    raw_message = '{"asr_reply": "Hello world how are you"}'
    initial_size = instance.message_buffer.qsize()

    instance._handle_asr_message(raw_message)

    final_size = instance.message_buffer.qsize()
    assert final_size == initial_size + 1
    assert instance.message_buffer.get_nowait() == "Hello world how are you"


def test_handle_asr_message_ignores_json_without_asr_reply(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    raw_message = '{"other_key": "other_value"}'
    initial_size = instance.message_buffer.qsize()

    instance._handle_asr_message(raw_message)

    final_size = instance.message_buffer.qsize()
    assert final_size == initial_size


def test_handle_asr_message_ignores_json_with_asr_reply_shorter_than_two_words(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    raw_message = '{"asr_reply": "Hi"}'
    initial_size = instance.message_buffer.qsize()

    instance._handle_asr_message(raw_message)

    final_size = instance.message_buffer.qsize()
    assert final_size == initial_size


def test_handle_asr_message_ignores_invalid_json(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    raw_message = "invalid json!"
    initial_size = instance.message_buffer.qsize()

    instance._handle_asr_message(raw_message)

    final_size = instance.message_buffer.qsize()
    assert final_size == initial_size


@pytest.mark.asyncio
async def test_raw_to_text_converts_string_to_message(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    test_data_str = "This is a test transcription."
    timestamp_before = time.time()

    result = await instance._raw_to_text(test_data_str)

    timestamp_after = time.time()
    assert result is not None
    assert result.message == test_data_str
    assert timestamp_before <= result.timestamp <= timestamp_after


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_if_input_none(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    result = await instance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_message_to_buffer(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    test_data_str = "First part of the message."
    initial_len = len(instance.messages)

    with patch("time.time", return_value=1234.0):
        await instance.raw_to_text(test_data_str)

    assert len(instance.messages) == initial_len + 1
    assert instance.messages[-1] == test_data_str


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_existing_message(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    instance.messages = ["Previous message"]

    test_data_str = "New part."
    await instance.raw_to_text(test_data_str)

    assert len(instance.messages) == 1
    assert instance.messages[-1] == "Previous message New part."


@pytest.mark.asyncio
async def test_raw_to_text_sets_skip_sleep_if_none_input_and_messages_exist(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    instance.messages = ["Existing message"]
    mock_sleep_ticker_instance.skip_sleep = False

    await instance.raw_to_text(None)

    assert mock_sleep_ticker_instance.skip_sleep is True


@pytest.mark.asyncio
async def test_raw_to_text_does_not_set_skip_sleep_if_none_input_and_messages_empty(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    instance.messages = []
    mock_sleep_ticker_instance.skip_sleep = False

    await instance.raw_to_text(None)

    assert mock_sleep_ticker_instance.skip_sleep is False


def test_formatted_latest_buffer_empty(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    result = instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_formats_and_clears_latest_message(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    fixed_timestamp = 1234.0
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    msg_content = "Final transcribed message."
    instance.messages = [msg_content]

    with patch("time.time", return_value=fixed_timestamp):
        result = instance.formatted_latest_buffer()

    assert result is not None
    assert "Voice" in result
    assert msg_content in result
    assert len(instance.messages) == 0
    mock_io_provider.add_input.assert_called_once_with("Voice", msg_content, fixed_timestamp)
    mock_io_provider.add_mode_transition_input.assert_called_once_with(msg_content)
    mock_teleops_conv_instance.store_user_message.assert_called_once_with(msg_content)
    if instance.asr_publisher:
        mock_zenoh["asr_text_cls"].assert_called_once()
        mock_zenoh["publisher"].put.assert_called_once()


def test_stop_clears_buffers_and_stops_asr(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    instance.messages = ["test message"]
    instance.message_buffer.put_nowait("buffered message")

    instance.stop()

    assert instance._stopped is True
    assert len(instance.messages) == 0
    assert instance.message_buffer.empty()
    mock_asr_instance.unregister_message_callback.assert_called_once()
    mock_zenoh["publisher"].undeclare.assert_called_once()
    mock_zenoh["session"].close.assert_called_once()


def test_stop_handles_exceptions_gracefully(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    mock_asr_instance.unregister_message_callback.side_effect = Exception("Unregister failed")
    mock_asr_instance.stop.side_effect = Exception("Stop failed")
    mock_zenoh["publisher"].undeclare.side_effect = Exception("Undeclare failed")

    instance.stop()

    assert instance._stopped is True


@pytest.mark.asyncio
async def test_poll_returns_none_when_stopped(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    instance._stopped = True
    instance.message_buffer.put_nowait("test message")

    result = await instance._poll()

    assert result is None


def test_initialization_with_unsupported_language(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(language="unsupported_language")

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]
    assert call_kwargs["language_code"] == "en-US"


def test_initialization_with_supported_language_chinese(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(language="chinese")

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]
    assert call_kwargs["language_code"] == "cmn-Hans-CN"


def test_initialization_with_custom_base_url(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    custom_base_url = "wss://custom.domain.com/asr"
    config = GoogleASRRTSPSensorConfig(base_url=custom_base_url)

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]
    assert call_kwargs["ws_url"] == custom_base_url


def test_initialization_with_zenoh_failure(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            side_effect=Exception("Zenoh connection failed"),
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    assert instance.session is None
    assert instance.asr_publisher is None


def test_formatted_latest_buffer_handles_zenoh_publish_exception(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    msg_content = "Test message"
    instance.messages = [msg_content]
    mock_zenoh["publisher"].put.side_effect = Exception("Zenoh publish failed")

    with patch("time.time", return_value=1234.0):
        result = instance.formatted_latest_buffer()

    assert result is not None
    assert msg_content in result
    assert len(instance.messages) == 0


def test_formatted_latest_buffer_without_zenoh_publisher(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
):
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            side_effect=Exception("Zenoh failed"),
        ),
    ):
        instance = GoogleASRRTSPInput(config=config)

    msg_content = "Test message without zenoh"
    instance.messages = [msg_content]

    with patch("time.time", return_value=1234.0):
        result = instance.formatted_latest_buffer()

    assert result is not None
    assert msg_content in result
    assert len(instance.messages) == 0
    mock_io_provider.add_input.assert_called_once_with("Voice", msg_content, 1234.0)


def test_initialization_with_enable_tts_interrupt_true(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(enable_tts_interrupt=True)

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]
    assert call_kwargs["enable_tts_interrupt"] is True


def test_initialization_with_api_version_v1(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(api_version="v1", api_key="test_key")

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]
    assert call_kwargs["ws_url"] == "wss://api.openmind.com/api/core/google/asr/v1?api_key=test_key"


def test_initialization_with_api_version_v2(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(api_version="v2", api_key="test_key")

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]
    assert call_kwargs["ws_url"] == "wss://api.openmind.com/api/core/google/asr/v2?api_key=test_key"


def test_initialization_with_invalid_api_version_defaults_to_v2(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(api_version="v3", api_key="test_key")

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]
    assert call_kwargs["ws_url"] == "wss://api.openmind.com/api/core/google/asr/v2?api_key=test_key"


def test_initialization_with_alternative_languages_v1(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(
        api_version="v1",
        language="english",
        alternative_languages=["chinese", "spanish"],
    )

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]

    assert call_kwargs["language_code"] == "en-US"
    assert "cmn-Hans-CN" in call_kwargs["alternative_language_codes"]
    assert "es-ES" in call_kwargs["alternative_language_codes"]


def test_initialization_with_alternative_languages_v2_ignored(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    mock_asr_constructor, _ = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(
        api_version="v2",
        language="english",
        alternative_languages=["chinese", "spanish"],
    )

    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch("inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
    ):
        GoogleASRRTSPInput(config=config)

    mock_asr_constructor.assert_called_once()
    call_kwargs = mock_asr_constructor.call_args[1]

    assert call_kwargs["language_code"] == "en-US"
    assert call_kwargs["alternative_language_codes"] == []


def test_handle_asr_message_speech_start_sets_timer(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """speech_start event sets _speech_start_time."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.time.time", return_value=100.0),
    ):
        instance = GoogleASRRTSPInput(config=config)
        assert instance._speech_start_time is None

        instance._handle_asr_message('{"type": "speech_start"}')

        assert instance._speech_start_time == 100.0


def test_handle_asr_message_speech_end_records_duration_metrics(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """speech_end event records speech duration histogram and gauge."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(language="english", api_version="v2")
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_speech_duration") as mock_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_speech_duration_last") as mock_gauge,
        patch("inputs.plugins.google_asr_rtsp.time.time", return_value=105.0),
    ):
        instance = GoogleASRRTSPInput(config=config)
        instance._speech_start_time = 100.0

        instance._handle_asr_message('{"type": "speech_end"}')

        mock_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(5.0)
        mock_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(5.0)


def test_handle_asr_message_speech_end_no_op_without_start_time(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """speech_end event is ignored when _speech_start_time is None."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_speech_duration") as mock_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_speech_duration_last") as mock_gauge,
    ):
        instance = GoogleASRRTSPInput(config=config)

        instance._handle_asr_message('{"type": "speech_end"}')

        mock_hist.labels().observe.assert_not_called()
        mock_gauge.labels().set.assert_not_called()


def test_handle_asr_message_end_of_utterance_records_latency_metrics(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """end_of_utterance event records utterance latency histogram and gauge."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(language="english", api_version="v2")
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_utterance_end_latency") as mock_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_utterance_end_latency_last") as mock_gauge,
        patch("inputs.plugins.google_asr_rtsp.time.time", return_value=103.5),
    ):
        instance = GoogleASRRTSPInput(config=config)
        instance._speech_start_time = 100.0

        instance._handle_asr_message('{"type": "end_of_utterance"}')

        mock_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(3.5)
        mock_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(3.5)


def test_handle_asr_message_end_of_utterance_no_op_without_start_time(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """end_of_utterance event is ignored when _speech_start_time is None."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_utterance_end_latency") as mock_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_utterance_end_latency_last") as mock_gauge,
    ):
        instance = GoogleASRRTSPInput(config=config)

        instance._handle_asr_message('{"type": "end_of_utterance"}')

        mock_hist.labels().observe.assert_not_called()
        mock_gauge.labels().set.assert_not_called()


def test_handle_asr_message_asr_reply_records_latency_and_resets_timer(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """asr_reply with speech_start set records ASR latency and resets _speech_start_time."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(language="english", api_version="v2")
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency") as mock_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency_last") as mock_gauge,
        patch("inputs.plugins.google_asr_rtsp.time.time", return_value=102.0),
    ):
        instance = GoogleASRRTSPInput(config=config)
        instance._speech_start_time = 100.0

        instance._handle_asr_message('{"asr_reply": "hello world"}')

        mock_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(2.0)
        mock_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(2.0)
        assert instance._speech_start_time is None


def test_handle_asr_message_asr_reply_no_latency_without_start_time(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """asr_reply without prior speech_start does not record latency metrics."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig()
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency") as mock_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency_last") as mock_gauge,
    ):
        instance = GoogleASRRTSPInput(config=config)

        instance._handle_asr_message('{"asr_reply": "hello world"}')

        mock_hist.labels().observe.assert_not_called()
        mock_gauge.labels().set.assert_not_called()


def test_handle_asr_message_full_sequence_records_all_metrics(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """Full speech_start -> speech_end -> asr_reply sequence records all three metric pairs."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(language="english", api_version="v2")
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_speech_duration") as mock_dur_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_speech_duration_last") as mock_dur_gauge,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency") as mock_lat_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency_last") as mock_lat_gauge,
    ):
        instance = GoogleASRRTSPInput(config=config)
        instance._speech_start_time = None

        with patch("inputs.plugins.google_asr_rtsp.time.time", return_value=100.0):
            instance._handle_asr_message('{"type": "speech_start"}')
        assert instance._speech_start_time == 100.0

        with patch("inputs.plugins.google_asr_rtsp.time.time", return_value=104.0):
            instance._handle_asr_message('{"type": "speech_end"}')
        mock_dur_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(4.0)
        mock_dur_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(4.0)

        with patch("inputs.plugins.google_asr_rtsp.time.time", return_value=106.0):
            instance._handle_asr_message('{"asr_reply": "hello world"}')
        mock_lat_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(6.0)
        mock_lat_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(6.0)
        assert instance._speech_start_time is None


def test_handle_asr_message_end_of_utterance_and_asr_reply_sequence(
    mock_io_provider,
    mock_asr_provider,
    mock_sleep_ticker_provider,
    mock_teleops_conversation_provider,
    mock_zenoh,
):
    """speech_start -> end_of_utterance -> asr_reply sequence records utterance and ASR latency."""
    _, mock_asr_instance = mock_asr_provider
    _, mock_sleep_ticker_instance = mock_sleep_ticker_provider
    _, mock_teleops_conv_instance = mock_teleops_conversation_provider

    config = GoogleASRRTSPSensorConfig(language="english", api_version="v2")
    with (
        patch("inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider",
            return_value=mock_asr_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.SleepTickerProvider",
            return_value=mock_sleep_ticker_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.TeleopsConversationProvider",
            return_value=mock_teleops_conv_instance,
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.open_zenoh_session",
            mock_zenoh["open_session"],
        ),
        patch("inputs.plugins.google_asr_rtsp.om1_asr_utterance_end_latency") as mock_utt_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_utterance_end_latency_last") as mock_utt_gauge,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency") as mock_lat_hist,
        patch("inputs.plugins.google_asr_rtsp.om1_asr_latency_last") as mock_lat_gauge,
    ):
        instance = GoogleASRRTSPInput(config=config)
        instance._speech_start_time = None

        with patch("inputs.plugins.google_asr_rtsp.time.time", return_value=100.0):
            instance._handle_asr_message('{"type": "speech_start"}')
        with patch("inputs.plugins.google_asr_rtsp.time.time", return_value=101.5):
            instance._handle_asr_message('{"type": "end_of_utterance"}')
        with patch("inputs.plugins.google_asr_rtsp.time.time", return_value=103.0):
            instance._handle_asr_message('{"asr_reply": "hello world"}')

        mock_utt_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(1.5)
        mock_utt_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(1.5)
        mock_lat_hist.labels(language="english", api_version="v2").observe.assert_called_once_with(3.0)
        mock_lat_gauge.labels(language="english", api_version="v2").set.assert_called_once_with(3.0)
