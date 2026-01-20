import time
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from providers.teleops_status_provider import (
    ActionStatus,
    ActionType,
    BatteryStatus,
    CommandStatus,
    TeleopsStatus,
    TeleopsStatusProvider,
)


def test_battery_status_roundtrip() -> None:
    ts = "2024-01-01T00:00:00Z"
    original = BatteryStatus(
        battery_level=85.5,
        temperature=42.0,
        voltage=12.3,
        timestamp=ts,
        charging_status=True,
    )

    as_dict = original.to_dict()
    restored = BatteryStatus.from_dict(as_dict)

    assert restored.battery_level == 85.5
    assert restored.temperature == 42.0
    assert restored.voltage == 12.3
    assert restored.timestamp == ts
    assert restored.charging_status is True


def test_command_status_roundtrip() -> None:
    ts = "2024-01-01T01:23:45Z"
    original = CommandStatus(vx=1.0, vy=-0.5, vyaw=0.25, timestamp=ts)

    as_dict = original.to_dict()
    restored = CommandStatus.from_dict(as_dict)

    assert restored.vx == 1.0
    assert restored.vy == -0.5
    assert restored.vyaw == 0.25
    assert restored.timestamp == ts


def test_action_status_roundtrip() -> None:
    now = time.time()
    original = ActionStatus(action=ActionType.TELEOPS, timestamp=now)

    as_dict = original.to_dict()
    restored = ActionStatus.from_dict(as_dict)

    assert restored.action == ActionType.TELEOPS
    assert isinstance(restored.timestamp, float)


def test_teleops_status_roundtrip() -> None:
    battery = BatteryStatus(
        battery_level=90.0,
        temperature=40.0,
        voltage=12.0,
        timestamp="2024-01-01T02:00:00Z",
        charging_status=False,
    )
    action = ActionStatus(action=ActionType.AI, timestamp=time.time())

    original = TeleopsStatus(
        update_time="2024-01-01T02:00:00Z",
        battery_status=battery,
        action_status=action,
        machine_name="robot-1",
        video_connected=True,
    )

    as_dict = original.to_dict()
    restored = TeleopsStatus.from_dict(as_dict)

    assert restored.machine_name == "robot-1"
    assert restored.video_connected is True
    assert isinstance(restored.battery_status, BatteryStatus)
    assert isinstance(restored.action_status, ActionStatus)


def reset_teleops_status_provider() -> None:
    """Reset singleton instance between tests."""
    TeleopsStatusProvider.reset()  # type: ignore[attr-defined]


@pytest.fixture
def provider() -> Generator[TeleopsStatusProvider, None, None]:
    reset_teleops_status_provider()
    instance = TeleopsStatusProvider(api_key="test-key", base_url="https://example.com")
    yield instance
    reset_teleops_status_provider()


@patch("providers.teleops_status_provider.requests.get")
def test_get_status_returns_empty_when_api_key_missing(mock_get: MagicMock) -> None:
    reset_teleops_status_provider()
    provider = TeleopsStatusProvider(api_key=None)

    result = provider.get_status()

    assert result == {}
    mock_get.assert_not_called()


@patch("providers.teleops_status_provider.requests.get")
def test_get_status_success(mock_get: MagicMock, provider: TeleopsStatusProvider) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"ok": True}
    mock_get.return_value = response

    result = provider.get_status()

    assert result == {"ok": True}
    mock_get.assert_called_once()


@patch("providers.teleops_status_provider.requests.get")
def test_get_status_non_200_returns_empty(
    mock_get: MagicMock, provider: TeleopsStatusProvider
) -> None:
    response = MagicMock()
    response.status_code = 500
    response.text = "error"
    mock_get.return_value = response

    result = provider.get_status()

    assert result == {}
    mock_get.assert_called_once()


@patch("providers.teleops_status_provider.requests.post")
def test_share_status_worker_skips_when_api_key_missing(
    mock_post: MagicMock,
) -> None:
    reset_teleops_status_provider()
    provider = TeleopsStatusProvider(api_key=None)

    status = TeleopsStatus(
        update_time="2024-01-01T00:00:00Z",
        battery_status=BatteryStatus(
            battery_level=50.0,
            temperature=30.0,
            voltage=11.0,
            timestamp="2024-01-01T00:00:00Z",
        ),
    )

    provider._share_status_worker(status)  # type: ignore[attr-defined]
    mock_post.assert_not_called()


@patch("providers.teleops_status_provider.requests.post")
def test_share_status_worker_sends_request_on_valid_status(
    mock_post: MagicMock, provider: TeleopsStatusProvider
) -> None:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"ok": True}
    mock_post.return_value = response

    status = TeleopsStatus(
        update_time="2024-01-01T00:00:00Z",
        battery_status=BatteryStatus(
            battery_level=50.0,
            temperature=30.0,
            voltage=11.0,
            timestamp="2024-01-01T00:00:00Z",
        ),
    )

    provider._share_status_worker(status)  # type: ignore[attr-defined]

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == provider.base_url
    assert kwargs["headers"]["Authorization"] == f"Bearer {provider.api_key}"
    assert kwargs["json"] == status.to_dict()


@patch("providers.teleops_status_provider.requests.post")
def test_share_status_worker_handles_exception(
    mock_post: MagicMock, provider: TeleopsStatusProvider
) -> None:
    mock_post.side_effect = Exception("network error")

    status = TeleopsStatus(
        update_time="2024-01-01T00:00:00Z",
        battery_status=BatteryStatus(
            battery_level=50.0,
            temperature=30.0,
            voltage=11.0,
            timestamp="2024-01-01T00:00:00Z",
        ),
    )

    provider._share_status_worker(status)  # type: ignore[attr-defined]


def test_share_status_submits_to_executor(provider: TeleopsStatusProvider) -> None:
    status = TeleopsStatus(
        update_time="2024-01-01T00:00:00Z",
        battery_status=BatteryStatus(
            battery_level=50.0,
            temperature=30.0,
            voltage=11.0,
            timestamp="2024-01-01T00:00:00Z",
        ),
    )

    provider.executor = MagicMock()

    provider.share_status(status)

    provider.executor.submit.assert_called_once()
    submit_args, _ = provider.executor.submit.call_args
    assert submit_args[0] == provider._share_status_worker  # type: ignore[attr-defined]

