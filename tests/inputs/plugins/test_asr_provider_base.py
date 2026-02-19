import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from inputs.asr_provider_base import BaseASRFuserInput
from inputs.base import SensorConfig


class DummyASRConfig(SensorConfig):
    """Minimal config for testing the base class."""

    api_key: str = "test-key"


class DummyASRInput(BaseASRFuserInput[DummyASRConfig]):
    """Concrete subclass for testing BaseASRFuserInput."""

    def __init__(self, config: DummyASRConfig):
        super().__init__(config)
        self.asr = Mock()


@pytest.fixture
def mock_base_deps():
    """Patch all base class dependencies."""
    with (
        patch("inputs.asr_provider_base.IOProvider") as mock_io,
        patch("inputs.asr_provider_base.SleepTickerProvider") as mock_sleep,
        patch(
            "inputs.asr_provider_base.TeleopsConversationProvider"
        ) as mock_conversation,
        patch("inputs.asr_provider_base.open_zenoh_session") as mock_zenoh,
    ):
        mock_session = Mock()
        mock_publisher = Mock()
        mock_zenoh.return_value = mock_session
        mock_session.declare_publisher.return_value = mock_publisher

        yield {
            "io": mock_io,
            "sleep_ticker": mock_sleep,
            "conversation": mock_conversation,
            "zenoh": mock_zenoh,
            "session": mock_session,
            "publisher": mock_publisher,
        }


def test_init_sets_up_common_attributes(mock_base_deps):
    """Test that __init__ sets up all common attributes."""
    config = DummyASRConfig()
    instance = DummyASRInput(config=config)

    assert isinstance(instance.messages, list)
    assert len(instance.messages) == 0
    assert instance.descriptor_for_LLM == "Voice"
    assert isinstance(instance.message_buffer, asyncio.Queue)
    assert instance.asr_topic == "om/asr/text"

    mock_base_deps["conversation"].assert_called_once_with(api_key="test-key")
    mock_base_deps["sleep_ticker"].assert_called_once()
    mock_base_deps["zenoh"].assert_called_once()
    mock_base_deps["session"].declare_publisher.assert_called_once_with("om/asr/text")
    assert instance.session is mock_base_deps["session"]
    assert instance.asr_publisher is mock_base_deps["publisher"]


def test_init_handles_zenoh_failure():
    """Test that __init__ handles Zenoh initialization failure gracefully."""
    with (
        patch("inputs.asr_provider_base.IOProvider"),
        patch("inputs.asr_provider_base.SleepTickerProvider"),
        patch("inputs.asr_provider_base.TeleopsConversationProvider"),
        patch(
            "inputs.asr_provider_base.open_zenoh_session",
            side_effect=Exception("Zenoh unavailable"),
        ),
    ):
        config = DummyASRConfig()
        instance = DummyASRInput(config=config)

        assert instance.session is None
        assert instance.asr_publisher is None


def test_handle_asr_message_valid_multi_word(mock_base_deps):
    """Test _handle_asr_message with valid multi-word asr_reply."""
    instance = DummyASRInput(config=DummyASRConfig())

    instance._handle_asr_message('{"asr_reply": "hello world"}')

    assert instance.message_buffer.qsize() == 1
    assert instance.message_buffer.get_nowait() == "hello world"


def test_handle_asr_message_single_word_ignored(mock_base_deps):
    """Test _handle_asr_message ignores single-word asr_reply."""
    instance = DummyASRInput(config=DummyASRConfig())

    instance._handle_asr_message('{"asr_reply": "hi"}')

    assert instance.message_buffer.qsize() == 0


def test_handle_asr_message_no_asr_reply_key(mock_base_deps):
    """Test _handle_asr_message ignores JSON without asr_reply."""
    instance = DummyASRInput(config=DummyASRConfig())

    instance._handle_asr_message('{"other": "data"}')

    assert instance.message_buffer.qsize() == 0


def test_handle_asr_message_invalid_json(mock_base_deps):
    """Test _handle_asr_message ignores invalid JSON."""
    instance = DummyASRInput(config=DummyASRConfig())

    instance._handle_asr_message("not valid json")

    assert instance.message_buffer.qsize() == 0


@pytest.mark.asyncio
async def test_poll_returns_message(mock_base_deps):
    """Test _poll returns message from buffer."""
    instance = DummyASRInput(config=DummyASRConfig())
    instance.message_buffer.put_nowait("test message")

    result = await instance._poll()

    assert result == "test message"


@pytest.mark.asyncio
async def test_poll_returns_none_when_empty(mock_base_deps):
    """Test _poll returns None when buffer is empty."""
    instance = DummyASRInput(config=DummyASRConfig())

    with patch("inputs.asr_provider_base.asyncio.sleep", new=AsyncMock()):
        result = await instance._poll()

    assert result is None


@pytest.mark.asyncio
async def test_poll_sleeps_when_empty(mock_base_deps):
    """Test _poll calls asyncio.sleep(0.01) when buffer is empty."""
    instance = DummyASRInput(config=DummyASRConfig())

    with patch("inputs.asr_provider_base.asyncio.sleep") as mock_sleep:
        await instance._poll()
        mock_sleep.assert_called_once_with(0.01)


@pytest.mark.asyncio
async def test_raw_to_text_returns_message(mock_base_deps):
    """Test _raw_to_text converts string to Message."""
    instance = DummyASRInput(config=DummyASRConfig())

    before = time.time()
    result = await instance._raw_to_text("hello")
    after = time.time()

    assert result is not None
    assert result.message == "hello"
    assert before <= result.timestamp <= after


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_for_none(mock_base_deps):
    """Test _raw_to_text returns None for None input."""
    instance = DummyASRInput(config=DummyASRConfig())

    result = await instance._raw_to_text(None)

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_first_message(mock_base_deps):
    """Test raw_to_text appends first message to empty list."""
    instance = DummyASRInput(config=DummyASRConfig())

    await instance.raw_to_text("hello")

    assert len(instance.messages) == 1
    assert instance.messages[0] == "hello"


@pytest.mark.asyncio
async def test_raw_to_text_concatenates_messages(mock_base_deps):
    """Test raw_to_text concatenates subsequent messages."""
    instance = DummyASRInput(config=DummyASRConfig())

    await instance.raw_to_text("hello")
    await instance.raw_to_text("world")

    assert len(instance.messages) == 1
    assert instance.messages[0] == "hello world"


@pytest.mark.asyncio
async def test_raw_to_text_none_sets_skip_sleep(mock_base_deps):
    """Test raw_to_text with None sets skip_sleep when messages exist."""
    instance = DummyASRInput(config=DummyASRConfig())
    instance.messages = ["existing"]

    await instance.raw_to_text(None)

    assert instance.global_sleep_ticker_provider.skip_sleep is True


@pytest.mark.asyncio
async def test_raw_to_text_none_no_skip_sleep_when_empty(mock_base_deps):
    """Test raw_to_text with None does not set skip_sleep when messages empty."""
    instance = DummyASRInput(config=DummyASRConfig())
    instance.messages = []
    instance.global_sleep_ticker_provider.skip_sleep = False

    await instance.raw_to_text(None)

    assert instance.global_sleep_ticker_provider.skip_sleep is False


def test_formatted_latest_buffer_empty(mock_base_deps):
    """Test formatted_latest_buffer returns None when empty."""
    instance = DummyASRInput(config=DummyASRConfig())

    result = instance.formatted_latest_buffer()

    assert result is None


def test_formatted_latest_buffer_formats_and_clears(mock_base_deps):
    """Test formatted_latest_buffer formats output and clears buffer."""
    instance = DummyASRInput(config=DummyASRConfig())
    instance.messages = ["test message"]

    result = instance.formatted_latest_buffer()

    assert result is not None
    assert "INPUT: Voice" in result
    assert "test message" in result
    assert "// START" in result
    assert "// END" in result
    assert len(instance.messages) == 0


def test_formatted_latest_buffer_publishes_to_zenoh(mock_base_deps):
    """Test formatted_latest_buffer publishes to Zenoh."""
    instance = DummyASRInput(config=DummyASRConfig())
    instance.messages = ["test message"]

    instance.formatted_latest_buffer()

    mock_base_deps["publisher"].put.assert_called_once()


def test_stop_stops_asr_and_closes_session(mock_base_deps):
    """Test stop method stops ASR provider and closes Zenoh session."""
    instance = DummyASRInput(config=DummyASRConfig())

    instance.stop()

    instance.asr.stop.assert_called_once()
    mock_base_deps["session"].close.assert_called_once()


def test_stop_no_session(mock_base_deps):
    """Test stop method when session is None."""
    instance = DummyASRInput(config=DummyASRConfig())
    instance.session = None

    instance.stop()

    instance.asr.stop.assert_called_once()


def test_stop_no_asr(mock_base_deps):
    """Test stop method when asr is None."""
    instance = DummyASRInput(config=DummyASRConfig())
    instance.asr = None

    instance.stop()

    mock_base_deps["session"].close.assert_called_once()
