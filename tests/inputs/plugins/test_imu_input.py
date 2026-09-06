import asyncio
from unittest.mock import MagicMock, patch

import pytest
import serial

from inputs.plugins.imu_input import IMUConfig, IMUInput
from providers.imu_provider import IMUProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    IMUProvider.reset()  # type: ignore[attr-defined]
    yield
    IMUProvider.reset()  # type: ignore[attr-defined]


@pytest.fixture
def config():
    return IMUConfig(
        port="/dev/ttyUSB0",
        baudrate=115200,
        timeout=1.0,
        fall_threshold=45.0,
        impact_threshold=20.0,
        poll_interval=0.1,
    )


@pytest.fixture
def imu_input(config):
    with patch("inputs.plugins.imu_input.serial.Serial") as mock_serial:
        mock_serial.return_value = MagicMock()
        plugin = IMUInput(config)
        return plugin


def test_init_success(config):
    with patch("inputs.plugins.imu_input.serial.Serial") as mock_serial:
        mock_serial.return_value = MagicMock()
        plugin = IMUInput(config)
        assert plugin.ser is not None
        assert (
            plugin.descriptor_for_LLM
            == "IMU Sensor (Accelerometer, Gyroscope, Orientation)"
        )


def test_init_serial_failure(config):
    with patch(
        "inputs.plugins.imu_input.serial.Serial",
        side_effect=serial.SerialException("Port not found"),
    ):
        plugin = IMUInput(config)
        assert plugin.ser is None


def test_thresholds_applied(config):
    with patch("inputs.plugins.imu_input.serial.Serial"):
        plugin = IMUInput(config)
        assert plugin.imu_provider.fall_threshold == 45.0
        assert plugin.imu_provider.impact_threshold == 20.0


def test_poll_no_serial(config):
    with patch(
        "inputs.plugins.imu_input.serial.Serial",
        side_effect=serial.SerialException("fail"),
    ):
        plugin = IMUInput(config)
        result = asyncio.get_event_loop().run_until_complete(plugin._poll())
        assert result is None


def test_poll_empty_line(imu_input):
    imu_input.ser.readline.return_value = b""
    result = asyncio.get_event_loop().run_until_complete(imu_input._poll())
    assert result is None


def test_poll_valid_data(imu_input):
    imu_input.ser.readline.return_value = b'{"ax":0.1,"ay":0.2,"az":9.8,"gx":0.0,"gy":0.0,"gz":0.0,"roll":1.0,"pitch":2.0,"yaw":90.0}\n'
    result = asyncio.get_event_loop().run_until_complete(imu_input._poll())
    assert result is not None
    assert result["ax"] == 0.1
    assert result["roll"] == 1.0


def test_poll_invalid_json(imu_input):
    imu_input.ser.readline.return_value = b"not json\n"
    result = asyncio.get_event_loop().run_until_complete(imu_input._poll())
    assert result is None


def test_raw_to_text_none(imu_input):
    result = asyncio.get_event_loop().run_until_complete(imu_input._raw_to_text(None))
    assert result is None


def test_raw_to_text_normal(imu_input):
    data = {
        "ax": 0.1,
        "ay": 0.2,
        "az": 9.8,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "roll": 1.0,
        "pitch": 2.0,
        "yaw": 90.0,
    }
    result = asyncio.get_event_loop().run_until_complete(imu_input._raw_to_text(data))
    assert result is not None
    assert "normal" in result.message.lower()


def test_raw_to_text_fall(imu_input):
    data = {
        "ax": 0.0,
        "ay": 0.0,
        "az": 9.8,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "roll": 50.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    result = asyncio.get_event_loop().run_until_complete(imu_input._raw_to_text(data))
    assert result is not None
    assert "fallen" in result.message.lower()


def test_raw_to_text_impact(imu_input):
    data = {
        "ax": 15.0,
        "ay": 15.0,
        "az": 0.0,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    result = asyncio.get_event_loop().run_until_complete(imu_input._raw_to_text(data))
    assert result is not None
    assert "impact" in result.message.lower()


def test_raw_to_text_invalid_data(imu_input):
    result = asyncio.get_event_loop().run_until_complete(
        imu_input._raw_to_text({"ax": "invalid"})
    )
    assert result is None


def test_raw_to_text_updates_buffer(imu_input):
    data = {
        "ax": 0.1,
        "ay": 0.2,
        "az": 9.8,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "roll": 1.0,
        "pitch": 2.0,
        "yaw": 90.0,
    }
    asyncio.get_event_loop().run_until_complete(imu_input.raw_to_text(data))
    assert len(imu_input.messages) == 1


def test_raw_to_text_none_not_added_to_buffer(imu_input):
    asyncio.get_event_loop().run_until_complete(imu_input.raw_to_text(None))
    assert len(imu_input.messages) == 0


def test_formatted_latest_buffer_empty(imu_input):
    result = imu_input.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_data(imu_input):
    data = {
        "ax": 0.1,
        "ay": 0.2,
        "az": 9.8,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "roll": 1.0,
        "pitch": 2.0,
        "yaw": 90.0,
    }
    asyncio.get_event_loop().run_until_complete(imu_input.raw_to_text(data))
    result = imu_input.formatted_latest_buffer()
    assert result is not None
    assert "IMU Sensor" in result
    assert len(imu_input.messages) == 0


def test_formatted_latest_buffer_clears_messages(imu_input):
    data = {
        "ax": 0.1,
        "ay": 0.2,
        "az": 9.8,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 0.0,
        "roll": 1.0,
        "pitch": 2.0,
        "yaw": 90.0,
    }
    asyncio.get_event_loop().run_until_complete(imu_input.raw_to_text(data))
    asyncio.get_event_loop().run_until_complete(imu_input.raw_to_text(data))
    imu_input.formatted_latest_buffer()
    assert len(imu_input.messages) == 0
