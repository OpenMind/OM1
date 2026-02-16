from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_initialization():
    """Test basic initialization."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        from inputs.plugins.turtlebot4_battery import (
            TurtleBot4Battery,
            TurtleBot4BatteryConfig,
        )

        config = TurtleBot4BatteryConfig()
        sensor = TurtleBot4Battery(config=config)

        assert sensor.messages == []
        assert sensor.battery_percentage == 0.0
        assert sensor.battery_voltage == 0.0
        assert sensor.is_docked is False


def test_initialization_with_custom_urid():
    """Test initialization with custom URID."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        from inputs.plugins.turtlebot4_battery import (
            TurtleBot4Battery,
            TurtleBot4BatteryConfig,
        )

        config = TurtleBot4BatteryConfig(URID="custom_robot")
        sensor = TurtleBot4Battery(config=config)

        assert sensor.URID == "custom_robot"


def test_listener_battery():
    """Test battery listener callback."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        from inputs.plugins.turtlebot4_battery import (
            TurtleBot4Battery,
            TurtleBot4BatteryConfig,
        )

        config = TurtleBot4BatteryConfig()
        sensor = TurtleBot4Battery(config=config)

        mock_sample = MagicMock()
        mock_msg = MagicMock()
        mock_msg.percentage = 0.855  # Will be converted to int(0.855 * 100) = 85
        mock_msg.voltage = 12.3
        mock_msg.temperature = 25.7
        mock_msg.header.stamp.sec = 1000

        with patch("inputs.plugins.turtlebot4_battery.sensor_msgs") as mock_sensor:
            mock_sensor.BatteryState.deserialize.return_value = mock_msg
            sensor.listener_battery(mock_sample)

        assert sensor.battery_percentage == 85  # int(0.855 * 100)
        assert sensor.battery_voltage == 12.3
        assert sensor.battery_temperature == 25.7


@pytest.mark.asyncio
async def test_poll():
    """Test _poll method."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        from inputs.plugins.turtlebot4_battery import (
            TurtleBot4Battery,
            TurtleBot4BatteryConfig,
        )

        config = TurtleBot4BatteryConfig()
        sensor = TurtleBot4Battery(config=config)
        sensor.battery_percentage = 10  # Low battery
        sensor.battery_voltage = 11.5
        sensor.is_docked = True
        sensor.battery_status = "IMPORTANT: your battery is low. Consider finding your charging station and recharging."

        with patch("inputs.plugins.turtlebot4_battery.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()

        assert result is not None
        assert len(result) == 1
        assert "battery" in result[0].lower()


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_data():
    """Test _raw_to_text with valid data."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        from inputs.plugins.turtlebot4_battery import (
            TurtleBot4Battery,
            TurtleBot4BatteryConfig,
        )

        config = TurtleBot4BatteryConfig()
        sensor = TurtleBot4Battery(config=config)

        with patch("inputs.plugins.turtlebot4_battery.time.time", return_value=1234.0):
            result = await sensor._raw_to_text(["75.5", "12.0", "True"])

        assert result is not None
        assert result.timestamp == 1234.0
        assert "75.5" in result.message or "75" in result.message


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer with messages."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        from inputs.base import Message
        from inputs.plugins.turtlebot4_battery import (
            TurtleBot4Battery,
            TurtleBot4BatteryConfig,
        )

        config = TurtleBot4BatteryConfig()
        sensor = TurtleBot4Battery(config=config)
        sensor.io_provider = MagicMock()

        sensor.messages = [
            Message(timestamp=1000.0, message="Battery: 80%"),
        ]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "Battery" in result or "battery" in result.lower()
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer with empty buffer."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        from inputs.plugins.turtlebot4_battery import (
            TurtleBot4Battery,
            TurtleBot4BatteryConfig,
        )

        config = TurtleBot4BatteryConfig()
        sensor = TurtleBot4Battery(config=config)

        result = sensor.formatted_latest_buffer()
        assert result is None
