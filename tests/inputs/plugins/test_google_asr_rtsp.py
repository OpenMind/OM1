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
    mock_sleep_ticker_constructor, mock_sleep_ticker_instance = (
        mock_sleep_ticker_provider
    )
    mock_teleops_conv_constructor, mock_teleops_conv_instance = (
        mock_teleops_conversation_provider
    )

    config = GoogleASRRTSPSensorConfig()
    api_key = config.api_key
    rtsp_url = config.rtsp_url
    rate = config.rate
    enable_tts_interrupt = config.enable_tts_interrupt

    with (
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
        patch(
            "inputs.plugins.google_asr_rtsp.ASRRTSPProvider", new=mock_asr_constructor
        ),
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
        ws_url=f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}",
        language_code="en-US",
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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

    with patch("asyncio.sleep") as mock_sleep:
        await instance._poll()
        mock_sleep.assert_called_once_with(0.1)


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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
        patch(
            "inputs.plugins.google_asr_rtsp.IOProvider", return_value=mock_io_provider
        ),
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
    assert "INPUT: Voice" in result
    assert msg_content in result
    assert len(instance.messages) == 0
    mock_io_provider.add_input.assert_called_once_with(
        "Voice", msg_content, fixed_timestamp
    )
    mock_io_provider.add_mode_transition_input.assert_called_once_with(msg_content)
    mock_teleops_conv_instance.store_user_message.assert_called_once_with(msg_content)
    if instance.asr_publisher:
        mock_zenoh["asr_text_cls"].assert_called_once()
        mock_zenoh["publisher"].put.assert_called_once()
