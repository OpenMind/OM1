from unittest.mock import patch

import pytest

from providers.teleops_status_provider import (
    ActionStatus,
    ActionType,
    BatteryStatus,
    CommandStatus,
    TeleopsStatus,
    TeleopsStatusProvider,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TeleopsStatusProvider.reset()  # type: ignore
    yield

    try:
        provider = TeleopsStatusProvider()
        provider.stop()
    except Exception:
        pass

    TeleopsStatusProvider.reset()  # type: ignore


def test_battery_status_creation():
    """Test BatteryStatus creation."""
    battery = BatteryStatus(
        battery_level=85.5,
        temperature=25.0,
        voltage=12.5,
        timestamp="2024-01-01T00:00:00",
        charging_status=True,
    )

    assert battery.battery_level == 85.5
    assert battery.temperature == 25.0
    assert battery.voltage == 12.5
    assert battery.timestamp == "2024-01-01T00:00:00"
    assert battery.charging_status is True


def test_battery_status_to_dict():
    """Test BatteryStatus to_dict conversion."""
    battery = BatteryStatus(
        battery_level=85.5,
        temperature=25.0,
        voltage=12.5,
        timestamp="2024-01-01T00:00:00",
        charging_status=False,
    )

    result = battery.to_dict()

    assert result["battery_level"] == 85.5
    assert result["temperature"] == 25.0
    assert result["voltage"] == 12.5
    assert result["timestamp"] == "2024-01-01T00:00:00"
    assert result["charging_status"] is False


def test_battery_status_from_dict():
    """Test BatteryStatus from_dict creation."""
    data = {
        "battery_level": 75.0,
        "temperature": 30.0,
        "voltage": 11.8,
        "timestamp": "2024-01-01T12:00:00",
        "charging_status": True,
    }

    battery = BatteryStatus.from_dict(data)

    assert battery.battery_level == 75.0
    assert battery.temperature == 30.0
    assert battery.voltage == 11.8
    assert battery.timestamp == "2024-01-01T12:00:00"
    assert battery.charging_status is True


def test_battery_status_from_dict_with_defaults():
    """Test BatteryStatus from_dict with missing fields."""
    data = {}
    battery = BatteryStatus.from_dict(data)

    assert battery.battery_level == 0.0
    assert battery.temperature == 0.0
    assert battery.voltage == 0.0
    assert battery.charging_status is False
    assert isinstance(battery.timestamp, str)


def test_command_status_creation():
    """Test CommandStatus creation."""
    command = CommandStatus(vx=1.5, vy=0.5, vyaw=0.2, timestamp="2024-01-01T00:00:00")

    assert command.vx == 1.5
    assert command.vy == 0.5
    assert command.vyaw == 0.2
    assert command.timestamp == "2024-01-01T00:00:00"


def test_command_status_to_dict():
    """Test CommandStatus to_dict conversion."""
    command = CommandStatus(vx=1.0, vy=0.0, vyaw=0.5, timestamp="2024-01-01T00:00:00")

    result = command.to_dict()

    assert result["vx"] == 1.0
    assert result["vy"] == 0.0
    assert result["vyaw"] == 0.5
    assert result["timestamp"] == "2024-01-01T00:00:00"


def test_command_status_from_dict():
    """Test CommandStatus from_dict creation."""
    data = {"vx": 2.0, "vy": 1.0, "vyaw": 0.3, "timestamp": "2024-01-01T12:00:00"}

    command = CommandStatus.from_dict(data)

    assert command.vx == 2.0
    assert command.vy == 1.0
    assert command.vyaw == 0.3
    assert command.timestamp == "2024-01-01T12:00:00"


def test_action_type_enum():
    """Test ActionType enum values."""
    assert ActionType.AI.value == "AI"
    assert ActionType.TELEOPS.value == "TELEOPS"
    assert ActionType.CONTROLLER.value == "CONTROLLER"


def test_action_status_creation():
    """Test ActionStatus creation."""
    action = ActionStatus(action=ActionType.AI, timestamp=1234567890.0)

    assert action.action == ActionType.AI
    assert action.timestamp == 1234567890.0


def test_action_status_to_dict():
    """Test ActionStatus to_dict conversion."""
    action = ActionStatus(action=ActionType.TELEOPS, timestamp=1234567890.0)

    result = action.to_dict()

    assert result["action"] == "TELEOPS"
    assert result["timestamp"] == 1234567890.0


def test_action_status_from_dict():
    """Test ActionStatus from_dict creation."""
    data = {"action": "CONTROLLER", "timestamp": 1234567890.0}

    action = ActionStatus.from_dict(data)

    assert action.action == ActionType.CONTROLLER
    assert action.timestamp == 1234567890.0


def test_teleops_status_creation():
    """Test TeleopsStatus creation."""
    battery = BatteryStatus(
        battery_level=80.0,
        temperature=25.0,
        voltage=12.0,
        timestamp="2024-01-01T00:00:00",
    )

    action = ActionStatus(action=ActionType.AI, timestamp=1234567890.0)

    status = TeleopsStatus(
        update_time="2024-01-01T00:00:00",
        battery_status=battery,
        action_status=action,
        machine_name="robot1",
        video_connected=True,
    )

    assert status.update_time == "2024-01-01T00:00:00"
    assert status.battery_status == battery
    assert status.action_status == action
    assert status.machine_name == "robot1"
    assert status.video_connected is True


def test_teleops_status_to_dict():
    """Test TeleopsStatus to_dict conversion."""
    battery = BatteryStatus(
        battery_level=80.0,
        temperature=25.0,
        voltage=12.0,
        timestamp="2024-01-01T00:00:00",
    )

    action = ActionStatus(action=ActionType.AI, timestamp=1234567890.0)

    status = TeleopsStatus(
        update_time="2024-01-01T00:00:00",
        battery_status=battery,
        action_status=action,
        machine_name="robot1",
        video_connected=False,
    )

    result = status.to_dict()

    assert result["update_time"] == "2024-01-01T00:00:00"
    assert result["machine_name"] == "robot1"
    assert result["video_connected"] is False
    assert isinstance(result["battery_status"], dict)
    assert isinstance(result["action_status"], dict)


def test_teleops_status_from_dict():
    """Test TeleopsStatus from_dict creation."""
    data = {
        "update_time": "2024-01-01T00:00:00",
        "machine_name": "robot2",
        "video_connected": True,
        "battery_status": {
            "battery_level": 90.0,
            "temperature": 20.0,
            "voltage": 12.5,
            "timestamp": "2024-01-01T00:00:00",
            "charging_status": True,
        },
        "action_status": {"action": "TELEOPS", "timestamp": 1234567890.0},
    }

    status = TeleopsStatus.from_dict(data)

    assert status.update_time == "2024-01-01T00:00:00"
    assert status.machine_name == "robot2"
    assert status.video_connected is True
    assert status.battery_status.battery_level == 90.0
    assert status.action_status.action == ActionType.TELEOPS


@pytest.fixture
def mock_teleops_dependencies():
    """Mock dependencies for TeleopsStatusProvider."""
    with (
        patch("providers.teleops_status_provider.requests.get") as mock_get,
        patch("providers.teleops_status_provider.requests.post") as mock_post,
    ):
        yield mock_get, mock_post


def test_teleops_status_provider_initialization(mock_teleops_dependencies):
    """Test TeleopsStatusProvider initialization."""
    provider = TeleopsStatusProvider(api_key="test_api_key_1234567890123456789")

    assert provider.api_key == "test_api_key_1234567890123456789"
    assert provider.base_url == "https://api.openmind.org/api/core/teleops/status"
    assert provider.executor is not None


def test_teleops_status_provider_singleton(mock_teleops_dependencies):
    """Test that TeleopsStatusProvider follows singleton pattern."""
    provider1 = TeleopsStatusProvider(api_key="key1")
    provider2 = TeleopsStatusProvider(api_key="key2")
    assert provider1 is provider2


def test_teleops_status_provider_initialization_failure():
    """Test handling of initialization failure."""
    provider = TeleopsStatusProvider()

    assert provider.api_key is None
    assert provider.base_url == "https://api.openmind.org/api/core/teleops/status"
