from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.turtlebot4_odom import Turtlebot4Odom, Turtlebot4OdomConfig


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        assert sensor.messages == []
        assert (
            "location" in sensor.descriptor_for_LLM.lower()
            or "pose" in sensor.descriptor_for_LLM.lower()
        )


def test_initialization_with_urid():
    """Test initialization with URID."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider") as mock_provider,
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig(URID="test_robot_123")
        Turtlebot4Odom(config=config)

        mock_provider.assert_called_once_with("test_robot_123")


def test_initialization_with_none_urid():
    """Test initialization with None URID."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider") as mock_provider,
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig(URID=None)
        Turtlebot4Odom(config=config)

        mock_provider.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_poll_with_position_data():
    """Test _poll with position data available."""
    with (
        patch(
            "inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class,
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = {"x": 1.0, "y": 2.0, "moving": False}
        mock_provider_class.return_value = mock_provider

        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        with patch("inputs.plugins.turtlebot4_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result == {"x": 1.0, "y": 2.0, "moving": False}


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no position data available."""
    with (
        patch(
            "inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"
        ) as mock_provider_class,
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = None
        mock_provider_class.return_value = mock_provider

        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        with patch("inputs.plugins.turtlebot4_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_moving_true():
    """Test _raw_to_text when robot is moving."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        raw_input = {"moving": True}
        message = await sensor._raw_to_text(raw_input)

        assert message is not None
        assert isinstance(message, Message)
        assert "moving" in message.message.lower()
        assert "do not generate" in message.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_with_moving_false():
    """Test _raw_to_text when robot is standing still."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        raw_input = {"moving": False}
        message = await sensor._raw_to_text(raw_input)

        assert message is not None
        assert isinstance(message, Message)
        assert "standing still" in message.message.lower()
        assert "can move" in message.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        message = await sensor._raw_to_text(None)

        assert message is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_messages():
    """Test raw_to_text appends messages to buffer."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        raw_input = {"moving": True}
        await sensor.raw_to_text(raw_input)

        assert len(sensor.messages) == 1
        assert isinstance(sensor.messages[0], Message)


@pytest.mark.asyncio
async def test_raw_to_text_with_none_does_not_append():
    """Test raw_to_text with None does not append to messages."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        await sensor.raw_to_text(None)

        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages in buffer."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider") as mock_io_provider_class,
    ):
        mock_io_provider = MagicMock()
        mock_io_provider_class.return_value = mock_io_provider

        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        # Add a message
        sensor.messages.append(Message(timestamp=123.456, message="Test message"))

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Test message" in result
        assert "INPUT:" in result
        assert "START" in result
        assert "END" in result
        assert len(sensor.messages) == 0  # Buffer should be cleared
        mock_io_provider.add_input.assert_called_once()


def test_formatted_latest_buffer_with_empty_buffer():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider"),
    ):
        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None


def test_formatted_latest_buffer_returns_latest_only():
    """Test formatted_latest_buffer returns only the latest message."""
    with (
        patch("inputs.plugins.turtlebot4_odom.TurtleBot4OdomProvider"),
        patch("inputs.plugins.turtlebot4_odom.IOProvider") as mock_io_provider_class,
    ):
        mock_io_provider = MagicMock()
        mock_io_provider_class.return_value = mock_io_provider

        config = Turtlebot4OdomConfig()
        sensor = Turtlebot4Odom(config=config)

        # Add multiple messages
        sensor.messages.append(Message(timestamp=123.0, message="First message"))
        sensor.messages.append(Message(timestamp=124.0, message="Second message"))
        sensor.messages.append(Message(timestamp=125.0, message="Third message"))

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Third message" in result
        assert "First message" not in result
        assert "Second message" not in result


def test_config_default_values():
    """Test default configuration values."""
    config = Turtlebot4OdomConfig()
    assert config.URID is None


def test_config_custom_urid():
    """Test custom URID configuration."""
    config = Turtlebot4OdomConfig(URID="robot_abc_123")
    assert config.URID == "robot_abc_123"
