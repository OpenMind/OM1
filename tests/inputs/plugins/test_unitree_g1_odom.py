from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_g1_odom import UnitreeG1Odom, UnitreeG1OdomConfig
from providers.unitree_g1_odom_provider import RobotState


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        assert sensor.messages == []
        assert (
            "location" in sensor.descriptor_for_LLM.lower()
            or "pose" in sensor.descriptor_for_LLM.lower()
        )


def test_initialization_with_unitree_ethernet():
    """Test initialization with Unitree ethernet channel."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider") as mock_provider,
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig(unitree_ethernet="eth0")
        UnitreeG1Odom(config=config)

        mock_provider.assert_called_once_with("eth0")


def test_initialization_without_ethernet():
    """Test initialization without ethernet channel."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider") as mock_provider,
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig(unitree_ethernet=None)
        UnitreeG1Odom(config=config)

        mock_provider.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_poll_with_position_data():
    """Test _poll with position data available."""
    with (
        patch(
            "inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"
        ) as mock_provider_class,
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = {"x": 1.0, "y": 2.0, "z": 0.3}
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        with patch("inputs.plugins.unitree_g1_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result == {"x": 1.0, "y": 2.0, "z": 0.3}


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no position data available."""
    with (
        patch(
            "inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"
        ) as mock_provider_class,
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = None
        mock_provider_class.return_value = mock_provider

        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        with patch("inputs.plugins.unitree_g1_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_standing_still():
    """Test _raw_to_text when robot is standing still."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        position_data = {"moving": False, "body_attitude": RobotState.STANDING}

        with patch("inputs.plugins.unitree_g1_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "standing still" in result.message.lower()
        assert "can move" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_moving():
    """Test _raw_to_text when robot is moving."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        position_data = {"moving": True, "body_attitude": RobotState.STANDING}

        with patch("inputs.plugins.unitree_g1_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "moving" in result.message.lower()
        assert "do not generate new movement commands" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_sitting():
    """Test _raw_to_text when robot is sitting."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        position_data = {"moving": False, "body_attitude": RobotState.SITTING}

        with patch("inputs.plugins.unitree_g1_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "sitting" in result.message.lower()
        assert "do not generate new movement commands" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_appends_to_messages():
    """Test raw_to_text appends message to buffer."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        position_data = {"moving": False, "body_attitude": RobotState.STANDING}

        with patch("inputs.plugins.unitree_g1_odom.time.time", return_value=1234.0):
            await sensor.raw_to_text(position_data)

        assert len(sensor.messages) == 1
        assert sensor.messages[0].timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_with_none_input():
    """Test raw_to_text with None input doesn't append."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        await sensor.raw_to_text(None)

        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(
                timestamp=1000.0,
                message="You are standing still - you can move if you want to. ",
            ),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "INPUT:" in result
        assert "// START" in result
        assert "// END" in result
        assert "standing still" in result.lower()
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None


def test_formatted_latest_buffer_clears_messages():
    """Test formatted_latest_buffer clears messages after formatting."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)
        sensor.io_provider = MagicMock()

        # Add multiple messages
        sensor.messages = [
            Message(timestamp=1000.0, message="Message 1"),
            Message(timestamp=1001.0, message="Message 2"),
            Message(timestamp=1002.0, message="Message 3"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Message 3" in result
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_returns_latest():
    """Test formatted_latest_buffer returns only the latest message."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Old message"),
            Message(timestamp=2000.0, message="Latest message"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Latest message" in result
        assert "Old message" not in result


def test_descriptor_for_llm():
    """Test descriptor_for_LLM is set correctly."""
    with (
        patch("inputs.plugins.unitree_g1_odom.UnitreeG1OdomProvider"),
        patch("inputs.plugins.unitree_g1_odom.IOProvider"),
    ):
        config = UnitreeG1OdomConfig()
        sensor = UnitreeG1Odom(config=config)

        assert hasattr(sensor, "descriptor_for_LLM")
        assert isinstance(sensor.descriptor_for_LLM, str)
        assert len(sensor.descriptor_for_LLM) > 0
