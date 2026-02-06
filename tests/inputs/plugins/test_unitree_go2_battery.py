from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.unitree_go2_battery import (
    UnitreeGo2Battery,
    UnitreeGo2BatteryConfig,
)


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        assert sensor.messages == []
        assert sensor.battery_percentage == 0.0
        assert sensor.battery_voltage == 0.0
        assert sensor.battery_amperes == 0.0


def test_initialization_with_api_key():
    """Test initialization with API key."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig(api_key="test_key")
        sensor = UnitreeGo2Battery(config=config)

        assert sensor.config.api_key == "test_key"


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)
        sensor.battery_percentage = 80.0
        sensor.battery_voltage = 24.5
        sensor.battery_amperes = 2.5

        with patch("inputs.plugins.unitree_go2_battery.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None
        assert len(result) == 3
        assert result[0] == 80.0
        assert result[1] == 24.5
        assert result[2] == 2.5


@pytest.mark.asyncio
async def test_raw_to_text_with_low_battery():
    """Test _raw_to_text with low battery (warning level)."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([10.0, 25.0, 3.0])

        assert result is not None
        assert result.timestamp == 1234.0
        assert "WARNING" in result.message or "energy" in result.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_with_critical_battery():
    """Test _raw_to_text with critical battery level."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([5.0, 25.0, 3.0])

        assert result is not None
        assert result.timestamp == 1234.0
        assert "CRITICAL" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_normal_battery():
    """Test _raw_to_text with normal battery level (no message)."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        with patch("inputs.plugins.unitree_go2_battery.time.time", return_value=1234.0):
            result = await sensor._raw_to_text([85.0, 25.0, 3.0])

        assert result is None


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Battery: 85%"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.unitree_go2_battery.ChannelSubscriber"),
        patch("inputs.plugins.unitree_go2_battery.IOProvider"),
        patch("inputs.plugins.unitree_go2_battery.TeleopsStatusProvider"),
    ):
        config = UnitreeGo2BatteryConfig()
        sensor = UnitreeGo2Battery(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
