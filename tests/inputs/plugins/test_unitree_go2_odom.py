from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_odom import UnitreeGo2Odom, UnitreeGo2OdomConfig
from providers.unitree_go2_odom_provider import RobotState


def test_initialization():
    """Test basic initialization with default use_sim=False."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider") as mock_cyclone_provider,
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider") as mock_zenoh_provider,
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        assert sensor.messages == []
        assert "location" in sensor.descriptor_for_LLM.lower() or "pose" in sensor.descriptor_for_LLM.lower()
        mock_cyclone_provider.assert_called_once_with(None)
        mock_zenoh_provider.assert_not_called()


def test_initialization_with_use_sim_true():
    """Test initialization with use_sim=True uses Zenoh provider."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider") as mock_cyclone_provider,
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider") as mock_zenoh_provider,
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig(use_sim=True, api_key="test_key", topic="test/topic")
        sensor = UnitreeGo2Odom(config=config)

        assert sensor.messages == []
        mock_zenoh_provider.assert_called_once_with(
            api_key="test_key",
            topic="test/topic",
            use_sim=True,
        )
        mock_cyclone_provider.assert_not_called()


def test_initialization_with_unitree_ethernet():
    """Test initialization with Unitree ethernet channel uses CycloneDDS provider."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider") as mock_cyclone_provider,
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider") as mock_zenoh_provider,
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig(unitree_ethernet="eth0")
        UnitreeGo2Odom(config=config)

        # Should use CycloneDDS provider when use_sim=False (default)
        mock_cyclone_provider.assert_called_once_with("eth0")
        mock_zenoh_provider.assert_not_called()


@pytest.mark.asyncio
async def test_poll_with_position_data():
    """Test _poll with position data available using CycloneDDS provider."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider") as mock_provider_class,
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = {"x": 1.0, "y": 2.0, "z": 0.0}
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        with patch("inputs.plugins.unitree_go2_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result == {"x": 1.0, "y": 2.0, "z": 0.0}


@pytest.mark.asyncio
async def test_poll_with_no_data():
    """Test _poll when no position data available."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider") as mock_provider_class,
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        mock_provider = MagicMock()
        mock_provider.position = None
        mock_provider_class.return_value = mock_provider

        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        with patch("inputs.plugins.unitree_go2_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test _raw_to_text with valid position data."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"),
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        position_data = {"moving": False, "body_attitude": RobotState.STANDING}

        with patch("inputs.plugins.unitree_go2_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert result.timestamp == 1234.0
        assert "standing still" in result.message.lower() or "can move" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test _raw_to_text with None input."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"),
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):

        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        result = await sensor._raw_to_text(None)
        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"),
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Position: x=1.0, y=2.0"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Position" in result or "position" in result.lower()
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"),
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None


@pytest.mark.asyncio
async def test_poll_with_zenoh_provider():
    """Test _poll with Zenoh provider when use_sim=True."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"),
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider") as mock_zenoh_class,
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        mock_zenoh = MagicMock()
        mock_zenoh.position = {"x": 5.0, "y": 10.0, "z": 0.5}
        mock_zenoh_class.return_value = mock_zenoh

        config = UnitreeGo2OdomConfig(use_sim=True, api_key="test_key")
        sensor = UnitreeGo2Odom(config=config)

        with patch("inputs.plugins.unitree_go2_odom.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result == {"x": 5.0, "y": 10.0, "z": 0.5}


@pytest.mark.asyncio
async def test_raw_to_text_sitting():
    """Test _raw_to_text when robot is sitting."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"),
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        position_data = {"moving": False, "body_attitude": RobotState.SITTING}

        with patch("inputs.plugins.unitree_go2_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert "sitting" in result.message.lower()
        assert "do not generate new movement" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_moving():
    """Test _raw_to_text when robot is moving."""
    with (
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomProvider"),
        patch("inputs.plugins.unitree_go2_odom.UnitreeGo2OdomZenohProvider"),
        patch("inputs.plugins.unitree_go2_odom.IOProvider"),
    ):
        config = UnitreeGo2OdomConfig()
        sensor = UnitreeGo2Odom(config=config)

        position_data = {"moving": True, "body_attitude": RobotState.STANDING}

        with patch("inputs.plugins.unitree_go2_odom.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(position_data)

        assert result is not None
        assert "moving" in result.message.lower()
        assert "do not generate new movement" in result.message.lower()
