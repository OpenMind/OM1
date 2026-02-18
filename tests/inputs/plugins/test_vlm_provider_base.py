from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message, SensorConfig
from inputs.vlm_provider_base import BaseVLMFuserInput


class DummyConfig(SensorConfig):
    pass


class DummyVLMPlugin(BaseVLMFuserInput[DummyConfig]):
    """Concrete subclass for testing BaseVLMFuserInput."""

    def __init__(self, config: DummyConfig):
        super().__init__(config)


def _create_sensor() -> DummyVLMPlugin:
    with patch("inputs.vlm_provider_base.IOProvider"):
        return DummyVLMPlugin(config=DummyConfig())


def test_initialization():
    """Test base class initialization sets up all shared state."""
    sensor = _create_sensor()

    assert sensor.messages == []
    assert sensor.message_buffer.empty()
    assert sensor.descriptor_for_LLM == "Vision"
    assert sensor.vlm is None


def test_descriptor_for_llm_class_override():
    """Test DESCRIPTOR_FOR_LLM can be overridden by subclass."""

    class CustomDescriptorPlugin(BaseVLMFuserInput[DummyConfig]):
        DESCRIPTOR_FOR_LLM = "CustomVision"

        def __init__(self, config: DummyConfig):
            super().__init__(config)

    with patch("inputs.vlm_provider_base.IOProvider"):
        sensor = CustomDescriptorPlugin(config=DummyConfig())

    assert sensor.descriptor_for_LLM == "CustomVision"


@pytest.mark.asyncio
async def test_poll_empty_buffer():
    """Test _poll returns None when buffer is empty."""
    with patch("inputs.vlm_provider_base.asyncio.sleep", new=AsyncMock()):
        sensor = _create_sensor()
        result = await sensor._poll()
        assert result is None


@pytest.mark.asyncio
async def test_poll_with_message():
    """Test _poll returns message when buffer has data."""
    with patch("inputs.vlm_provider_base.asyncio.sleep", new=AsyncMock()):
        sensor = _create_sensor()
        sensor.message_buffer.put("hello world")
        result = await sensor._poll()
        assert result == "hello world"
        assert sensor.message_buffer.empty()


@pytest.mark.asyncio
async def test_raw_to_text_none():
    """Test _raw_to_text returns None for None input."""
    sensor = _create_sensor()
    result = await sensor._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_valid():
    """Test _raw_to_text creates a Message with timestamp."""
    sensor = _create_sensor()
    result = await sensor._raw_to_text("test message")
    assert isinstance(result, Message)
    assert result.message == "test message"
    assert result.timestamp > 0


@pytest.mark.asyncio
async def test_raw_to_text_public_none():
    """Test raw_to_text does nothing for None input."""
    sensor = _create_sensor()
    await sensor.raw_to_text(None)
    assert len(sensor.messages) == 0


@pytest.mark.asyncio
async def test_raw_to_text_public_valid():
    """Test raw_to_text appends message to buffer."""
    sensor = _create_sensor()
    await sensor.raw_to_text("test message")
    assert len(sensor.messages) == 1
    assert sensor.messages[0].message == "test message"


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when no messages."""
    sensor = _create_sensor()
    result = sensor.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_message():
    """Test formatted_latest_buffer formats and clears messages."""
    sensor = _create_sensor()
    sensor.messages.append(Message(timestamp=100.0, message="I see a robot"))

    result = sensor.formatted_latest_buffer()

    assert isinstance(result, str)
    assert "INPUT: Vision" in result
    assert "// START" in result
    assert "I see a robot" in result
    assert "// END" in result
    assert len(sensor.messages) == 0

    sensor.io_provider.add_input.assert_called_once_with(
        "DummyVLMPlugin", "I see a robot", 100.0
    )


def test_stop_with_vlm():
    """Test stop calls vlm.stop() when vlm is set."""
    sensor = _create_sensor()
    mock_vlm = MagicMock()
    sensor.vlm = mock_vlm

    sensor.stop()
    mock_vlm.stop.assert_called_once()


def test_stop_without_vlm():
    """Test stop is safe when vlm is None."""
    sensor = _create_sensor()
    sensor.stop()
