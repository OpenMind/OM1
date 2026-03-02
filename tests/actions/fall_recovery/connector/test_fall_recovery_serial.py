import asyncio
from unittest.mock import MagicMock, patch

import pytest

from actions.fall_recovery.connector.serial import (
    FallRecoverySerialConfig,
    FallRecoverySerialConnector,
)
from actions.fall_recovery.interface import FallRecoveryAction, FallRecoveryInput
from providers.imu_provider import IMUProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    IMUProvider.reset()  # type: ignore[attr-defined]
    yield
    IMUProvider.reset()  # type: ignore[attr-defined]


@pytest.fixture
def config():
    return FallRecoverySerialConfig(
        port="/dev/ttyUSB0",
        baudrate=115200,
        timeout=2.0,
    )


@pytest.fixture
def connector(config):
    with patch(
        "actions.fall_recovery.connector.serial._pyserial.Serial"
    ) as mock_serial:
        mock_serial.return_value = MagicMock()
        c = FallRecoverySerialConnector(config)
        return c


def test_init_success(config):
    with patch(
        "actions.fall_recovery.connector.serial._pyserial.Serial"
    ) as mock_serial:
        mock_serial.return_value = MagicMock()
        c = FallRecoverySerialConnector(config)
        assert c.ser is not None


def test_init_serial_failure(config):
    with patch(
        "actions.fall_recovery.connector.serial._pyserial.Serial",
        side_effect=Exception("Port not found"),
    ):
        c = FallRecoverySerialConnector(config)
        assert c.ser is None


def test_send_command_success(connector):
    result = connector._send_command({"action": "stand_up"})
    assert result is True
    connector.ser.write.assert_called_once()


def test_send_command_no_serial(config):
    with patch(
        "actions.fall_recovery.connector.serial._pyserial.Serial",
        side_effect=Exception("fail"),
    ):
        c = FallRecoverySerialConnector(config)
        result = c._send_command({"action": "stand_up"})
        assert result is False


def test_send_command_write_error(connector):
    connector.ser.write.side_effect = Exception("write error")
    result = connector._send_command({"action": "stand_up"})
    assert result is False


def test_connect_stand_up(connector):
    output = FallRecoveryInput(action=FallRecoveryAction.STAND_UP, message="Robot fell")
    asyncio.get_event_loop().run_until_complete(connector.connect(output))
    connector.ser.write.assert_called_once()
    written = connector.ser.write.call_args[0][0].decode("utf-8")
    assert "stand_up" in written


def test_connect_stand_up_resets_alerts(connector):
    IMUProvider().update(15.0, 15.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)
    assert IMUProvider().state["is_fallen"] is True
    output = FallRecoveryInput(action=FallRecoveryAction.STAND_UP, message="")
    asyncio.get_event_loop().run_until_complete(connector.connect(output))
    assert IMUProvider().state["is_fallen"] is False


def test_connect_emergency_stop(connector):
    output = FallRecoveryInput(
        action=FallRecoveryAction.EMERGENCY_STOP, message="Critical impact"
    )
    asyncio.get_event_loop().run_until_complete(connector.connect(output))
    connector.ser.write.assert_called_once()
    written = connector.ser.write.call_args[0][0].decode("utf-8")
    assert "emergency_stop" in written


def test_connect_alert_operator(connector):
    output = FallRecoveryInput(
        action=FallRecoveryAction.ALERT_OPERATOR, message="Need help"
    )
    asyncio.get_event_loop().run_until_complete(connector.connect(output))
    connector.ser.write.assert_called_once()
    written = connector.ser.write.call_args[0][0].decode("utf-8")
    assert "alert_operator" in written


def test_connect_unknown_action(connector):
    output = FallRecoveryInput(action=FallRecoveryAction.STAND_UP, message="")
    output.action = MagicMock()
    output.action.value = "unknown_action"
    asyncio.get_event_loop().run_until_complete(connector.connect(output))
    connector.ser.write.assert_not_called()


def test_send_command_json_format(connector):
    connector._send_command({"action": "stand_up", "message": "test"})
    written = connector.ser.write.call_args[0][0].decode("utf-8")
    import json

    data = json.loads(written.strip())
    assert data["action"] == "stand_up"
    assert data["message"] == "test"
