"""Tests for TeleopsStatusProvider and related dataclasses."""

import logging
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Mock external dependencies before imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["rclpy"] = MagicMock()
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()


class TestBatteryStatus:
    """Tests for BatteryStatus dataclass."""

    def test_battery_status_creation(self):
        """Test BatteryStatus can be created with all fields."""
        from providers.teleops_status_provider import BatteryStatus

        status = BatteryStatus(
            battery_level=85.5,
            temperature=25.0,
            voltage=12.6,
            timestamp="2024-01-01T00:00:00",
            charging_status=True,
        )

        assert status.battery_level == 85.5
        assert status.temperature == 25.0
        assert status.voltage == 12.6
        assert status.timestamp == "2024-01-01T00:00:00"
        assert status.charging_status is True

    def test_battery_status_to_dict(self):
        """Test BatteryStatus to_dict method."""
        from providers.teleops_status_provider import BatteryStatus

        status = BatteryStatus(
            battery_level=75.0,
            temperature=30.0,
            voltage=11.8,
            timestamp="2024-01-01T12:00:00",
            charging_status=False,
        )

        result = status.to_dict()

        assert result["battery_level"] == 75.0
        assert result["temperature"] == 30.0
        assert result["voltage"] == 11.8
        assert result["timestamp"] == "2024-01-01T12:00:00"
        assert result["charging_status"] is False

    def test_battery_status_from_dict(self):
        """Test BatteryStatus from_dict class method."""
        from providers.teleops_status_provider import BatteryStatus

        data = {
            "battery_level": 90.0,
            "temperature": 22.5,
            "voltage": 12.8,
            "timestamp": "2024-01-02T00:00:00",
            "charging_status": True,
        }

        status = BatteryStatus.from_dict(data)

        assert status.battery_level == 90.0
        assert status.temperature == 22.5
        assert status.voltage == 12.8
        assert status.timestamp == "2024-01-02T00:00:00"
        assert status.charging_status is True

    def test_battery_status_from_dict_defaults(self):
        """Test BatteryStatus from_dict with missing fields uses defaults."""
        from providers.teleops_status_provider import BatteryStatus

        status = BatteryStatus.from_dict({})

        assert status.battery_level == 0.0
        assert status.temperature == 0.0
        assert status.voltage == 0.0
        assert status.charging_status is False


class TestCommandStatus:
    """Tests for CommandStatus dataclass."""

    def test_command_status_creation(self):
        """Test CommandStatus can be created with all fields."""
        from providers.teleops_status_provider import CommandStatus

        status = CommandStatus(
            vx=1.5,
            vy=0.5,
            vyaw=0.2,
            timestamp="2024-01-01T00:00:00",
        )

        assert status.vx == 1.5
        assert status.vy == 0.5
        assert status.vyaw == 0.2
        assert status.timestamp == "2024-01-01T00:00:00"

    def test_command_status_to_dict(self):
        """Test CommandStatus to_dict method."""
        from providers.teleops_status_provider import CommandStatus

        status = CommandStatus(
            vx=2.0, vy=-1.0, vyaw=0.5, timestamp="2024-01-01T00:00:00"
        )

        result = status.to_dict()

        assert result["vx"] == 2.0
        assert result["vy"] == -1.0
        assert result["vyaw"] == 0.5

    def test_command_status_from_dict(self):
        """Test CommandStatus from_dict class method."""
        from providers.teleops_status_provider import CommandStatus

        data = {"vx": 1.0, "vy": 2.0, "vyaw": 0.3, "timestamp": "2024-01-01T00:00:00"}

        status = CommandStatus.from_dict(data)

        assert status.vx == 1.0
        assert status.vy == 2.0
        assert status.vyaw == 0.3


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_type_values(self):
        """Test ActionType enum has correct values."""
        from providers.teleops_status_provider import ActionType

        assert ActionType.AI.value == "AI"
        assert ActionType.TELEOPS.value == "TELEOPS"
        assert ActionType.CONTROLLER.value == "CONTROLLER"


class TestActionStatus:
    """Tests for ActionStatus dataclass."""

    def test_action_status_creation(self):
        """Test ActionStatus can be created."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        status = ActionStatus(action=ActionType.TELEOPS, timestamp=1234567890.0)

        assert status.action == ActionType.TELEOPS
        assert status.timestamp == 1234567890.0

    def test_action_status_to_dict(self):
        """Test ActionStatus to_dict method."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        status = ActionStatus(action=ActionType.AI, timestamp=1234567890.0)

        result = status.to_dict()

        assert result["action"] == "AI"
        assert result["timestamp"] == 1234567890.0

    def test_action_status_from_dict(self):
        """Test ActionStatus from_dict class method."""
        from providers.teleops_status_provider import ActionStatus, ActionType

        data = {"action": "CONTROLLER", "timestamp": 9876543210.0}

        status = ActionStatus.from_dict(data)

        assert status.action == ActionType.CONTROLLER
        assert status.timestamp == 9876543210.0


class TestTeleopsStatus:
    """Tests for TeleopsStatus dataclass."""

    def test_teleops_status_creation(self):
        """Test TeleopsStatus can be created."""
        from providers.teleops_status_provider import (
            ActionStatus,
            ActionType,
            BatteryStatus,
            TeleopsStatus,
        )

        battery = BatteryStatus(
            battery_level=80.0,
            temperature=25.0,
            voltage=12.0,
            timestamp="2024-01-01T00:00:00",
        )
        action = ActionStatus(action=ActionType.AI, timestamp=time.time())

        status = TeleopsStatus(
            update_time="2024-01-01T00:00:00",
            battery_status=battery,
            action_status=action,
            machine_name="test-robot",
            video_connected=True,
        )

        assert status.machine_name == "test-robot"
        assert status.video_connected is True
        assert status.battery_status.battery_level == 80.0

    def test_teleops_status_to_dict(self):
        """Test TeleopsStatus to_dict method."""
        from providers.teleops_status_provider import (
            ActionStatus,
            ActionType,
            BatteryStatus,
            TeleopsStatus,
        )

        battery = BatteryStatus(
            battery_level=70.0,
            temperature=28.0,
            voltage=11.5,
            timestamp="2024-01-01T00:00:00",
        )
        action = ActionStatus(action=ActionType.TELEOPS, timestamp=1234567890.0)

        status = TeleopsStatus(
            update_time="2024-01-01T12:00:00",
            battery_status=battery,
            action_status=action,
            machine_name="robot-1",
            video_connected=False,
        )

        result = status.to_dict()

        assert result["machine_name"] == "robot-1"
        assert result["video_connected"] is False
        assert result["battery_status"]["battery_level"] == 70.0
        assert result["action_status"]["action"] == "TELEOPS"

    def test_teleops_status_from_dict(self):
        """Test TeleopsStatus from_dict class method."""
        from providers.teleops_status_provider import ActionType, TeleopsStatus

        data = {
            "update_time": "2024-01-01T00:00:00",
            "machine_name": "robot-2",
            "video_connected": True,
            "battery_status": {
                "battery_level": 95.0,
                "temperature": 20.0,
                "voltage": 13.0,
                "timestamp": "2024-01-01T00:00:00",
                "charging_status": True,
            },
            "action_status": {"action": "AI", "timestamp": 1234567890.0},
        }

        status = TeleopsStatus.from_dict(data)

        assert status.machine_name == "robot-2"
        assert status.video_connected is True
        assert status.battery_status.battery_level == 95.0
        assert status.action_status.action == ActionType.AI


class TestTeleopsStatusProvider:
    """Tests for TeleopsStatusProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Clear cached modules before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith("providers")]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith("providers")]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_provider_initialization(self):
        """Test TeleopsStatusProvider initialization."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider(api_key="test-api-key")

        assert provider.api_key == "test-api-key"
        assert provider.base_url == "https://api.openmind.org/api/core/teleops/status"

    def test_provider_initialization_custom_url(self):
        """Test TeleopsStatusProvider with custom base URL."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider(
            api_key="test-key", base_url="https://custom.api.com/status"
        )

        assert provider.base_url == "https://custom.api.com/status"

    @patch("providers.teleops_status_provider.requests.get")
    def test_get_status_success(self, mock_get):
        """Test successful status retrieval."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "battery": 80}
        mock_get.return_value = mock_response

        provider = TeleopsStatusProvider(api_key="sk-1234567890123456789012345")

        result = provider.get_status()

        assert result == {"status": "ok", "battery": 80}
        mock_get.assert_called_once()

    @patch("providers.teleops_status_provider.requests.get")
    def test_get_status_no_api_key(self, mock_get, caplog):
        """Test get_status with missing API key."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider(api_key=None)

        with caplog.at_level(logging.ERROR):
            result = provider.get_status()

        assert result == {}
        mock_get.assert_not_called()

    @patch("providers.teleops_status_provider.requests.get")
    def test_get_status_api_error(self, mock_get, caplog):
        """Test get_status handles API errors gracefully."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        provider = TeleopsStatusProvider(api_key="sk-1234567890123456789012345")

        with caplog.at_level(logging.ERROR):
            result = provider.get_status()

        assert result == {}

    @patch("providers.teleops_status_provider.requests.get")
    def test_get_status_request_exception(self, mock_get, caplog):
        """Test get_status handles request exceptions."""
        import requests

        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        provider = TeleopsStatusProvider(api_key="sk-1234567890123456789012345")

        with caplog.at_level(logging.ERROR):
            result = provider.get_status()

        assert result == {}

    @patch("providers.teleops_status_provider.requests.post")
    def test_share_status_worker_success(self, mock_post):
        """Test successful status sharing."""
        from providers.teleops_status_provider import (
            BatteryStatus,
            TeleopsStatus,
            TeleopsStatusProvider,
        )

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        provider = TeleopsStatusProvider(api_key="test-api-key")

        battery = BatteryStatus(
            battery_level=80.0,
            temperature=25.0,
            voltage=12.0,
            timestamp="2024-01-01T00:00:00",
        )
        status = TeleopsStatus(
            update_time="2024-01-01T00:00:00", battery_status=battery
        )

        provider._share_status_worker(status)

        mock_post.assert_called_once()

    @patch("providers.teleops_status_provider.requests.post")
    def test_share_status_worker_no_api_key(self, mock_post, caplog):
        """Test share_status_worker with missing API key."""
        from providers.teleops_status_provider import (
            BatteryStatus,
            TeleopsStatus,
            TeleopsStatusProvider,
        )

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider = TeleopsStatusProvider(api_key=None)

        battery = BatteryStatus(
            battery_level=80.0,
            temperature=25.0,
            voltage=12.0,
            timestamp="2024-01-01T00:00:00",
        )
        status = TeleopsStatus(
            update_time="2024-01-01T00:00:00", battery_status=battery
        )

        with caplog.at_level(logging.ERROR):
            provider._share_status_worker(status)

        mock_post.assert_not_called()

    def test_singleton_behavior(self):
        """Test that TeleopsStatusProvider is a singleton."""
        from providers.teleops_status_provider import TeleopsStatusProvider

        if hasattr(TeleopsStatusProvider, "reset"):
            TeleopsStatusProvider.reset()

        provider1 = TeleopsStatusProvider(api_key="key1")
        provider2 = TeleopsStatusProvider(api_key="key2")

        assert provider1 is provider2
