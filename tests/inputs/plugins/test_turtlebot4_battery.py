from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.turtlebot4_battery import TurtleBot4Battery, TurtleBot4BatteryConfig


@pytest.fixture
def mock_deps():
    """Mock all external dependencies."""
    with (
        patch("inputs.plugins.turtlebot4_battery.open_zenoh_session"),
        patch("inputs.plugins.turtlebot4_battery.IOProvider"),
        patch("inputs.plugins.turtlebot4_battery.TeleopsStatusProvider"),
    ):
        yield


@pytest.fixture
def sensor(mock_deps):
    """Create a TurtleBot4Battery sensor with mocked dependencies."""
    config = TurtleBot4BatteryConfig()
    return TurtleBot4Battery(config=config)


def test_initialization(mock_deps):
    """Test basic initialization."""
    config = TurtleBot4BatteryConfig()
    sensor = TurtleBot4Battery(config=config)

    assert sensor.messages == []
    assert sensor.battery_percentage == 0.0
    assert sensor.battery_voltage == 0.0
    assert sensor.is_docked is False


def test_initialization_with_custom_urid(mock_deps):
    """Test initialization with custom URID."""
    config = TurtleBot4BatteryConfig(URID="custom_robot")
    sensor = TurtleBot4Battery(config=config)

    assert sensor.URID == "custom_robot"


def test_listener_battery(sensor):
    """Test battery listener callback."""
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


def test_listener_battery_critical(sensor):
    """Test battery listener callback with critical battery level."""
    mock_sample = MagicMock()
    mock_msg = MagicMock()
    mock_msg.percentage = 0.03
    mock_msg.voltage = 10.0
    mock_msg.temperature = 25.0
    mock_msg.header.stamp.sec = 1000

    with patch("inputs.plugins.turtlebot4_battery.sensor_msgs") as mock_sensor:
        mock_sensor.BatteryState.deserialize.return_value = mock_msg
        sensor.listener_battery(mock_sample)

    assert sensor.battery_percentage == 3
    assert "CRITICAL" in sensor.battery_status


def test_listener_battery_normal(sensor):
    """Test battery listener callback with normal battery level."""
    mock_sample = MagicMock()
    mock_msg = MagicMock()
    mock_msg.percentage = 0.80
    mock_msg.voltage = 12.5
    mock_msg.temperature = 25.0
    mock_msg.header.stamp.sec = 1000

    with patch("inputs.plugins.turtlebot4_battery.sensor_msgs") as mock_sensor:
        mock_sensor.BatteryState.deserialize.return_value = mock_msg
        sensor.listener_battery(mock_sample)

    assert sensor.battery_percentage == 80
    assert sensor.battery_status is None


def test_listener_dock(sensor):
    """Test dock status listener callback."""
    mock_sample = MagicMock()
    mock_dock = MagicMock()
    mock_dock.is_docked = True

    with patch("inputs.plugins.turtlebot4_battery.sensor_msgs") as mock_sensor:
        mock_sensor.DockStatus.deserialize.return_value = mock_dock
        sensor.listener_dock(mock_sample)

    assert sensor.is_docked is True


@pytest.mark.asyncio
async def test_poll_with_low_battery(sensor):
    """Test _poll method with low battery."""
    sensor.battery_percentage = 10
    sensor.battery_voltage = 11.5
    sensor.is_docked = True
    sensor.battery_status = "IMPORTANT: your battery is low. Consider finding your charging station and recharging."

    with patch("inputs.plugins.turtlebot4_battery.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert result is not None
    assert len(result) == 1
    assert "battery" in result[0].lower()


@pytest.mark.asyncio
async def test_poll_with_normal_battery(sensor):
    """Test _poll method with normal battery."""
    sensor.battery_percentage = 80
    sensor.battery_status = None

    with patch("inputs.plugins.turtlebot4_battery.asyncio.sleep", new=AsyncMock()):
        result = await sensor._poll()

    assert result == []


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_data(sensor):
    """Test _raw_to_text with valid data."""
    with patch("inputs.plugins.turtlebot4_battery.time.time", return_value=1234.0):
        result = await sensor._raw_to_text(["75.5", "12.0", "True"])

    assert result is not None
    assert result.timestamp == 1234.0
    assert "75.5" in result.message or "75" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_with_empty_data(sensor):
    """Test _raw_to_text with empty data."""
    result = await sensor._raw_to_text([])
    assert result is None


def test_formatted_latest_buffer_with_messages(sensor):
    """Test formatted_latest_buffer with messages."""
    sensor.io_provider = MagicMock()
    sensor.messages = [
        Message(timestamp=1000.0, message="Battery: 80%"),
    ]

    result = sensor.formatted_latest_buffer()

    assert result is not None
    assert "Battery" in result or "battery" in result.lower()
    sensor.io_provider.add_input.assert_called_once()
    assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty(sensor):
    """Test formatted_latest_buffer with empty buffer."""
    result = sensor.formatted_latest_buffer()
    assert result is None
