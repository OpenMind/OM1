from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.tron_odom import TronOdom, TronOdomConfig
from providers.tron_odom_provider import RobotState


def test_initialization():
    """Test basic initialization with default config."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider"),
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        assert sensor.messages == []
        assert (
            "location" in sensor.descriptor_for_LLM.lower()
            or "pose" in sensor.descriptor_for_LLM.lower()
        )


def test_initialization_with_custom_topic():
    """Test initialization with custom topic."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider") as mock_provider,
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig(topic="custom_odom")
        sensor = TronOdom(config=config)

        assert sensor.config.topic == "custom_odom"
        mock_provider.assert_called_once_with("custom_odom")


@pytest.mark.asyncio
async def test_poll_with_position_data():
    """Test _poll with position data available."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider") as mock_provider_class,
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = {"odom_x": 1.0, "odom_y": 2.0, "moving": False}
        mock_provider_class.return_value = mock_provider

        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        with patch("inputs.plugins.tron_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result == {"odom_x": 1.0, "odom_y": 2.0, "moving": False}


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no position data available."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider") as mock_provider_class,
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = None
        mock_provider_class.return_value = mock_provider

        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        with patch("inputs.plugins.tron_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_sitting():
    """Test _raw_to_text when robot is sitting."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider"),
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        position_data = {"moving": False, "body_attitude": RobotState.SITTING}

        with patch("inputs.plugins.tron_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "sitting" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_moving():
    """Test _raw_to_text when robot is moving."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider"),
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        position_data = {"moving": True, "body_attitude": RobotState.STANDING}

        with patch("inputs.plugins.tron_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "moving" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_standing_still():
    """Test _raw_to_text when robot is standing still."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider"),
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        position_data = {"moving": False, "body_attitude": RobotState.STANDING}

        with patch("inputs.plugins.tron_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert (
            "standing still" in result.message.lower()
            or "can move" in result.message.lower()
        )


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider"),
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider"),
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig()
        sensor = TronOdom(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="You are standing still"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "standing" in result.lower() or "INPUT" in result
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.tron_odom.TronOdomProvider"),
        patch("inputs.plugins.tron_odom.IOProvider"),
    ):
        config = TronOdomConfig()
        sensor = TronOdom(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
