import time
from unittest.mock import MagicMock, patch

from requests.exceptions import RequestException

from src.providers.teleops_status_provider import (
    ActionStatus,
    ActionType,
    BatteryStatus,
    CommandStatus,
    TeleopsStatus,
    TeleopsStatusProvider,
)

# --- Tests for Dataclasses and Enum ---


class TestBatteryStatus:
    def test_to_dict(self):
        status = BatteryStatus(
            battery_level=80.0,
            temperature=25.0,
            voltage=12.5,
            timestamp="2023-01-01T00:00:00Z",
            charging_status=True,
        )
        expected_dict = {
            "battery_level": 80.0,
            "charging_status": True,
            "temperature": 25.0,
            "voltage": 12.5,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        assert status.to_dict() == expected_dict

    def test_from_dict(self):
        input_dict = {
            "battery_level": 75.5,
            "charging_status": False,
            "temperature": 30.0,
            "voltage": 11.8,
            "timestamp": "2023-01-01T00:01:00Z",
        }
        status = BatteryStatus.from_dict(input_dict)
        assert status.battery_level == 75.5
        assert not status.charging_status
        assert status.temperature == 30.0
        assert status.voltage == 11.8
        assert status.timestamp == "2023-01-01T00:01:00Z"

    def test_from_dict_defaults(self):
        # Test defaults for missing fields
        input_dict = {}
        status = BatteryStatus.from_dict(input_dict)
        assert status.battery_level == 0.0
        assert not status.charging_status
        assert status.temperature == 0.0
        assert status.voltage == 0.0
        assert isinstance(status.timestamp, str)  # Default is str(time.time())


class TestCommandStatus:
    def test_to_dict(self):
        status = CommandStatus(
            vx=1.0, vy=0.0, vyaw=0.5, timestamp="2023-01-01T00:00:00Z"
        )
        expected_dict = {
            "vx": 1.0,
            "vy": 0.0,
            "vyaw": 0.5,
            "timestamp": "2023-01-01T00:00:00Z",
        }
        assert status.to_dict() == expected_dict

    def test_from_dict(self):
        input_dict = {
            "vx": 0.5,
            "vy": -0.2,
            "vyaw": 0.1,
            "timestamp": "2023-01-01T00:01:00Z",
        }
        status = CommandStatus.from_dict(input_dict)
        assert status.vx == 0.5
        assert status.vy == -0.2
        assert status.vyaw == 0.1
        assert status.timestamp == "2023-01-01T00:01:00Z"

    def test_from_dict_defaults(self):
        # Test defaults for missing fields
        input_dict = {}
        status = CommandStatus.from_dict(input_dict)
        assert status.vx == 0.0
        assert status.vy == 0.0
        assert status.vyaw == 0.0


class TestActionStatus:
    def test_to_dict(self):
        status = ActionStatus(action=ActionType.TELEOPS, timestamp=1234567890.0)
        expected_dict = {
            "action": "TELEOPS",  # ActionType.TELEOPS.value
            "timestamp": 1234567890.0,
        }
        assert status.to_dict() == expected_dict

    def test_from_dict(self):
        input_dict = {
            "action": "CONTROLLER",
            "timestamp": 1234567891.0,
        }
        status = ActionStatus.from_dict(input_dict)
        assert status.action == ActionType.CONTROLLER
        assert status.timestamp == 1234567891.0

    def test_from_dict_defaults(self):
        # Test defaults for missing fields
        input_dict = {}
        status = ActionStatus.from_dict(input_dict)
        assert status.action == ActionType.AI  # Default value


class TestTeleopsStatus:
    def test_to_dict(self):
        battery_status = BatteryStatus(
            battery_level=90.0, temperature=20.0, voltage=12.0, timestamp="now"
        )
        action_status = ActionStatus(action=ActionType.AI, timestamp=time.time())
        status = TeleopsStatus(
            update_time="2023-01-01T00:00:00Z",
            battery_status=battery_status,
            action_status=action_status,
            machine_name="robot1",
            video_connected=True,
        )
        expected_dict = {
            "machine_name": "robot1",
            "update_time": "2023-01-01T00:00:00Z",
            "battery_status": battery_status.to_dict(),
            "action_status": action_status.to_dict(),
            "video_connected": True,
        }
        assert status.to_dict() == expected_dict

    def test_from_dict(self):
        input_dict = {
            "machine_name": "robot2",
            "update_time": "2023-01-01T00:01:00Z",
            "battery_status": {
                "battery_level": 85.0,
                "temperature": 22.0,
                "voltage": 11.9,
                "timestamp": "then",
                "charging_status": False,
            },
            "action_status": {"action": "TELEOPS", "timestamp": 1234567892.0},
            "video_connected": False,
        }
        status = TeleopsStatus.from_dict(input_dict)
        assert status.machine_name == "robot2"
        assert status.update_time == "2023-01-01T00:01:00Z"
        assert status.battery_status.battery_level == 85.0
        assert status.action_status.action == ActionType.TELEOPS
        assert not status.video_connected


# --- Tests for TeleopsStatusProvider Class ---
# Note: TeleopsStatusProvider is a singleton. Mocking is required for isolation.


class TestTeleopsStatusProvider:

    @patch("src.providers.teleops_status_provider.requests.get")
    def test_get_status_success(self, mock_get, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        api_key = "valid_key_1234567890123456789012345"
        base_url = "https://test.api.com/"
        provider = TeleopsStatusProvider(api_key=api_key, base_url=base_url)
        provider.api_key = api_key
        provider.base_url = base_url

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response

        result = provider.get_status()

        expected_api_key_id = api_key[9:25] if len(api_key) > 25 else api_key
        expected_url = f"{provider.base_url}/{expected_api_key_id}"

        mock_get.assert_called_once_with(
            expected_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        assert result == {"status": "ok"}

    @patch("src.providers.teleops_status_provider.requests.get")
    def test_get_status_missing_api_key(self, mock_get, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        provider = TeleopsStatusProvider()
        provider.api_key = None
        provider.base_url = "https://test.api.com/"

        result = provider.get_status()

        mock_get.assert_not_called()
        assert result == {}

    @patch("src.providers.teleops_status_provider.requests.get")
    def test_get_status_request_exception(self, mock_get, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        api_key = "valid_key_1234567890123456789012345"
        base_url = "https://test.api.com/"
        provider = TeleopsStatusProvider(api_key=api_key, base_url=base_url)
        provider.api_key = api_key
        provider.base_url = base_url

        mock_get.side_effect = RequestException("Network Error")

        result = provider.get_status()

        assert result == {}

    @patch("src.providers.teleops_status_provider.requests.get")
    def test_get_status_non_200_response(self, mock_get, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        api_key = "valid_key_1234567890123456789012345"
        base_url = "https://test.api.com/"
        provider = TeleopsStatusProvider(api_key=api_key, base_url=base_url)
        provider.api_key = api_key
        provider.base_url = base_url

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        result = provider.get_status()

        assert result == {}

    @patch("src.providers.teleops_status_provider.requests.post")
    def test_share_status_worker_success(self, mock_post, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        api_key = "valid_key_1234567890123456789012345"
        base_url = "https://test.api.com/"
        provider = TeleopsStatusProvider(api_key=api_key, base_url=base_url)
        provider.api_key = api_key
        provider.base_url = base_url

        status_obj = TeleopsStatus(
            update_time="2023-01-01T00:00:00Z",
            battery_status=BatteryStatus(100.0, 25.0, 12.5, "now"),
            action_status=ActionStatus(ActionType.AI, time.time()),
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "success"}
        mock_post.return_value = mock_response

        provider._share_status_worker(status_obj)

        mock_post.assert_called_once_with(
            provider.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            json=status_obj.to_dict(),
            timeout=10,
        )

    @patch("src.providers.teleops_status_provider.requests.post")
    def test_share_status_worker_missing_api_key(self, mock_post, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        provider = TeleopsStatusProvider()
        provider.api_key = None
        provider.base_url = "https://test.api.com/"

        status_obj = TeleopsStatus(
            update_time="2023-01-01T00:00:00Z",
            battery_status=BatteryStatus(100.0, 25.0, 12.5, "now"),
            action_status=ActionStatus(ActionType.AI, time.time()),
        )

        provider._share_status_worker(status_obj)

        mock_post.assert_not_called()

    @patch("src.providers.teleops_status_provider.requests.post")
    def test_share_status_worker_request_exception(self, mock_post, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        api_key = "valid_key_1234567890123456789012345"
        base_url = "https://test.api.com/"
        provider = TeleopsStatusProvider(api_key=api_key, base_url=base_url)
        provider.api_key = api_key
        provider.base_url = base_url

        status_obj = TeleopsStatus(
            update_time="2023-01-01T00:00:00Z",
            battery_status=BatteryStatus(100.0, 25.0, 12.5, "now"),
            action_status=ActionStatus(ActionType.AI, time.time()),
        )

        mock_post.side_effect = RequestException("Network Error")

        provider._share_status_worker(status_obj)

    @patch("src.providers.teleops_status_provider.requests.post")
    def test_share_status_worker_non_200_response(self, mock_post, monkeypatch):
        # Mock the singleton instance
        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )

        api_key = "valid_key_1234567890123456789012345"
        base_url = "https://test.api.com/"
        provider = TeleopsStatusProvider(api_key=api_key, base_url=base_url)
        provider.api_key = api_key
        provider.base_url = base_url

        status_obj = TeleopsStatus(
            update_time="2023-01-01T00:00:00Z",
            battery_status=BatteryStatus(100.0, 25.0, 12.5, "now"),
            action_status=ActionStatus(ActionType.AI, time.time()),
        )

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        provider._share_status_worker(status_obj)

    @patch("src.providers.teleops_status_provider.ThreadPoolExecutor")
    def test_share_status_submits_task(self, mock_executor_class, monkeypatch):
        # Mock the singleton instance and its executor
        mock_executor_instance = MagicMock()
        mock_executor_class.return_value = mock_executor_instance

        mock_provider_instance = MagicMock()
        monkeypatch.setattr(
            "src.providers.teleops_status_provider.TeleopsStatusProvider",
            lambda *a, **kw: mock_provider_instance,
        )
        mock_provider_instance.executor = mock_executor_instance

        provider = TeleopsStatusProvider()

        status_obj = TeleopsStatus(
            update_time="2023-01-01T00:00:00Z",
            battery_status=BatteryStatus(100.0, 25.0, 12.5, "now"),
            action_status=ActionStatus(ActionType.AI, time.time()),
        )

        # Mock submit to verify it's called
        provider.executor.submit = MagicMock()

        provider.share_status(status_obj)

        # Verify submit was called with _share_status_worker method and status object
        from unittest.mock import ANY

        provider.executor.submit.assert_called_once_with(ANY, status_obj)
