from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.riva_asr import (
    RivaASRInput,
    RivaASRSensorConfig,
    _normalize_asr_text,
    _seems_directed_at_robot,
)


class TestNormalizeAsrText:
    """Tests for _normalize_asr_text."""

    def test_corrects_om_one_variants(self):
        assert "OM1" in _normalize_asr_text("tell me about om one")
        assert "OM1" in _normalize_asr_text("what is ol one")
        assert "OM1" in _normalize_asr_text("I like om 1")

    def test_corrects_open_mind(self):
        assert "OpenMind" in _normalize_asr_text("this is open mind")

    def test_corrects_unit_tree(self):
        assert "Unitree" in _normalize_asr_text("the unit tree robot")

    def test_no_correction_needed(self):
        assert _normalize_asr_text("hello world") == "hello world"


class TestSeemsDirectedAtRobot:
    """Tests for _seems_directed_at_robot."""

    def test_directed_keyword_match(self):
        assert _seems_directed_at_robot("hello how are you") is True

    def test_question_mark_detected(self):
        assert _seems_directed_at_robot("something random stuff?") is True

    def test_overheard_chatter_filtered(self):
        assert _seems_directed_at_robot("yeah totally agree man") is False

    def test_case_insensitive(self):
        assert _seems_directed_at_robot("HELLO there friend") is True


def test_handle_asr_message_filters_overheard_chatter():
    """Test that _handle_asr_message filters messages not directed at robot."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        # Message with >2 words but not directed at robot
        raw_message = '{"asr_reply": "yeah totally agree man"}'
        sensor._handle_asr_message(raw_message)
        assert sensor.message_buffer.qsize() == 0


def test_handle_asr_message_accepts_directed_speech():
    """Test that _handle_asr_message accepts messages directed at robot."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        raw_message = '{"asr_reply": "hello how are you"}'
        sensor._handle_asr_message(raw_message)
        assert sensor.message_buffer.qsize() == 1


def test_handle_asr_message_normalizes_text():
    """Test that _handle_asr_message applies ASR text normalization."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        raw_message = '{"asr_reply": "tell me about om one please"}'
        sensor._handle_asr_message(raw_message)
        assert sensor.message_buffer.qsize() == 1
        assert "OM1" in sensor.message_buffer.get_nowait()


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider") as mock_asr,
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):
        mock_asr_instance = MagicMock()
        mock_asr.return_value = mock_asr_instance

        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        assert hasattr(sensor, "messages")
        mock_asr_instance.start.assert_called_once()


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):
        config = RivaASRSensorConfig()
        sensor = RivaASRInput(config=config)

        with patch("inputs.plugins.riva_asr.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()
            assert result is None


def test_formatted_latest_buffer():
    """Test formatted_latest_buffer."""
    with (
        patch("inputs.plugins.riva_asr.IOProvider"),
        patch("inputs.plugins.riva_asr.ASRProvider"),
        patch("inputs.plugins.riva_asr.SleepTickerProvider"),
    ):
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
