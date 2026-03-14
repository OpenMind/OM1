"""
Tests for PIRMotionInput plugin.

Follows the OM1 test conventions from:
- tests/inputs/plugins/test_serial_reader.py
- tests/inputs/plugins/test_mock_input.py

Run with:
    pytest tests/inputs/plugins/test_pir_motion.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.base import Message
from inputs.plugins.pir_motion import (
    PIRMotionConfig,
    PIRMotionInput,
    _GPIOPIRConnector,
    _MockPIRConnector,
    _SerialPIRConnector,
    _ZenohPIRConnector,
)


def make_plugin(connector: str = "mock", cooldown: float = 0.0) -> PIRMotionInput:
    config = PIRMotionConfig(connector=connector, cooldown=cooldown)
    with patch("inputs.plugins.pir_motion.IOProvider"):
        return PIRMotionInput(config=config)


def test_default_connector_is_mock():
    config = PIRMotionConfig()
    assert config.connector == "mock"


def test_default_cooldown():
    config = PIRMotionConfig()
    assert config.cooldown == 5.0


def test_serial_config_fields():
    config = PIRMotionConfig(connector="serial", port="/dev/ttyUSB0", baudrate=115200)
    assert config.connector == "serial"
    assert config.port == "/dev/ttyUSB0"
    assert config.baudrate == 115200


def test_gpio_config_fields():
    config = PIRMotionConfig(connector="gpio", gpio_pin=27)
    assert config.gpio_pin == 27


def test_zenoh_config_fields():
    config = PIRMotionConfig(connector="zenoh", zenoh_topic="robot/pir")
    assert config.zenoh_topic == "robot/pir"


def test_mock_connector_selected_by_default():
    plugin = make_plugin("mock")
    assert isinstance(plugin._connector, _MockPIRConnector)


def test_unknown_connector_falls_back_to_mock():
    plugin = make_plugin("nonexistent_xyz")
    assert isinstance(plugin._connector, _MockPIRConnector)


def test_serial_connector_selected():
    with patch("inputs.plugins.pir_motion._serial.Serial"):
        plugin = make_plugin("serial")
    assert isinstance(plugin._connector, _SerialPIRConnector)


def test_gpio_connector_selected():
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.IN = 1
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._GPIO_LIB", mock_gpio),
    ):
        plugin = make_plugin("gpio")
    assert isinstance(plugin._connector, _GPIOPIRConnector)


def test_zenoh_connector_selected():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        plugin = make_plugin("zenoh")
    assert isinstance(plugin._connector, _ZenohPIRConnector)


@pytest.mark.asyncio
async def test_poll_returns_true_on_motion():
    plugin = make_plugin(cooldown=0.0)
    plugin._connector.read = AsyncMock(return_value=True)
    with patch("inputs.plugins.pir_motion.asyncio.sleep", new=AsyncMock()):
        result = await plugin._poll()
    assert result is True


@pytest.mark.asyncio
async def test_poll_returns_false_on_no_motion():
    plugin = make_plugin(cooldown=0.0)
    plugin._connector.read = AsyncMock(return_value=False)
    with patch("inputs.plugins.pir_motion.asyncio.sleep", new=AsyncMock()):
        result = await plugin._poll()
    assert result is False


@pytest.mark.asyncio
async def test_poll_returns_false_when_connector_returns_none():
    plugin = make_plugin(cooldown=0.0)
    plugin._connector.read = AsyncMock(return_value=None)
    with patch("inputs.plugins.pir_motion.asyncio.sleep", new=AsyncMock()):
        result = await plugin._poll()
    assert result is False


@pytest.mark.asyncio
async def test_cooldown_suppresses_second_motion():
    plugin = make_plugin(cooldown=60.0)
    plugin._connector.read = AsyncMock(return_value=True)
    with patch("inputs.plugins.pir_motion.asyncio.sleep", new=AsyncMock()):
        first = await plugin._poll()
        second = await plugin._poll()
    assert first is True
    assert second is None


@pytest.mark.asyncio
async def test_cooldown_allows_motion_after_elapsed():
    plugin = make_plugin(cooldown=0.0)
    plugin._connector.read = AsyncMock(return_value=True)
    with patch("inputs.plugins.pir_motion.asyncio.sleep", new=AsyncMock()):
        first = await plugin._poll()
        second = await plugin._poll()
    assert first is True
    assert second is True


@pytest.mark.asyncio
async def test_raw_to_text_true_produces_message():
    plugin = make_plugin()
    with patch("inputs.plugins.pir_motion.time.time", return_value=1234.0):
        msg = await plugin._raw_to_text(True)
    assert msg is not None
    assert msg.timestamp == 1234.0
    assert "motion" in msg.message.lower()


@pytest.mark.asyncio
async def test_raw_to_text_false_returns_none():
    plugin = make_plugin()
    msg = await plugin._raw_to_text(False)
    assert msg is None


@pytest.mark.asyncio
async def test_raw_to_text_none_returns_none():
    plugin = make_plugin()
    msg = await plugin._raw_to_text(None)
    assert msg is None


@pytest.mark.asyncio
async def test_raw_to_text_fills_buffer_on_motion():
    plugin = make_plugin()
    await plugin.raw_to_text(True)
    assert len(plugin.messages) == 1


@pytest.mark.asyncio
async def test_raw_to_text_does_not_fill_buffer_on_false():
    plugin = make_plugin()
    await plugin.raw_to_text(False)
    assert len(plugin.messages) == 0


def test_formatted_latest_buffer_empty_returns_none():
    plugin = make_plugin()
    assert plugin.formatted_latest_buffer() is None


def test_formatted_latest_buffer_returns_formatted_string():
    plugin = make_plugin()
    plugin.io_provider = MagicMock()
    plugin.messages = [Message(timestamp=1000.0, message="Motion detected.")]
    result = plugin.formatted_latest_buffer()
    assert result is not None
    assert "PIR Motion Sensor" in result
    assert "Motion detected." in result


def test_formatted_latest_buffer_clears_messages():
    plugin = make_plugin()
    plugin.io_provider = MagicMock()
    plugin.messages = [Message(timestamp=1000.0, message="Motion detected.")]
    plugin.formatted_latest_buffer()
    assert len(plugin.messages) == 0


def test_formatted_latest_buffer_calls_io_provider():
    plugin = make_plugin()
    plugin.io_provider = MagicMock()
    plugin.messages = [Message(timestamp=1000.0, message="Motion detected.")]
    plugin.formatted_latest_buffer()
    plugin.io_provider.add_input.assert_called_once()


def test_stop_calls_connector_stop():
    plugin = make_plugin()
    plugin._connector.stop = MagicMock()
    plugin.stop()
    plugin._connector.stop.assert_called_once()


def test_stop_clears_message_buffer():
    plugin = make_plugin()
    plugin.messages = [Message(timestamp=1000.0, message="test")]
    plugin.stop()
    assert plugin.messages == []


@pytest.mark.asyncio
async def test_mock_connector_returns_bool():
    connector = _MockPIRConnector(trigger_interval=1)
    result = await connector.read()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_mock_connector_eventually_triggers():
    connector = _MockPIRConnector(trigger_interval=3)
    results = [await connector.read() for _ in range(30)]
    assert True in results


def test_serial_connector_no_serial_lib():
    with patch("inputs.plugins.pir_motion._SERIAL_AVAILABLE", False):
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
    assert connector._ser is None


@pytest.mark.asyncio
async def test_serial_connector_parse_motion_1():
    with patch("inputs.plugins.pir_motion._serial.Serial") as mock_ser:
        mock_ser.return_value.readline.return_value = b"MOTION:1\r\n"
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
    assert result is True


@pytest.mark.asyncio
async def test_serial_connector_parse_motion_0():
    with patch("inputs.plugins.pir_motion._serial.Serial") as mock_ser:
        mock_ser.return_value.readline.return_value = b"MOTION:0\r\n"
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
    assert result is False


@pytest.mark.asyncio
async def test_serial_connector_unknown_line_returns_none():
    with patch("inputs.plugins.pir_motion._serial.Serial") as mock_ser:
        mock_ser.return_value.readline.return_value = b"GARBAGE\r\n"
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
    assert result is None


@pytest.mark.asyncio
async def test_serial_connector_no_connection_returns_none():
    with patch("inputs.plugins.pir_motion._SERIAL_AVAILABLE", False):
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
    assert result is None


def test_gpio_connector_no_lib():
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", False),
        patch("inputs.plugins.pir_motion._GPIO_LIB", None),
    ):
        connector = _GPIOPIRConnector(pin=17)
    assert connector._ready is False


@pytest.mark.asyncio
async def test_gpio_connector_reads_high():
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.IN = 1
    mock_gpio.input.return_value = 1
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._GPIO_LIB", mock_gpio),
    ):
        connector = _GPIOPIRConnector(pin=17)
        result = await connector.read()
    assert result is True


@pytest.mark.asyncio
async def test_gpio_connector_reads_low():
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.IN = 1
    mock_gpio.input.return_value = 0
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._GPIO_LIB", mock_gpio),
    ):
        connector = _GPIOPIRConnector(pin=17)
        result = await connector.read()
    assert result is False


@pytest.mark.asyncio
async def test_serial_connector_open_fails():
    import serial as _serial

    with patch(
        "inputs.plugins.pir_motion._serial.Serial",
        side_effect=_serial.SerialException("Port busy"),
    ):
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
    assert connector._ser is None


@pytest.mark.asyncio
async def test_serial_connector_read_exception_returns_none():
    with patch("inputs.plugins.pir_motion._serial.Serial") as mock_ser:
        mock_ser.return_value.readline.side_effect = Exception("read error")
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
    assert result is None


def test_gpio_connector_setup_exception():
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.IN = 1
    mock_gpio.setup.side_effect = Exception("GPIO busy")
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._GPIO_LIB", mock_gpio),
    ):
        connector = _GPIOPIRConnector(pin=17)
    assert connector._ready is False


@pytest.mark.asyncio
async def test_gpio_connector_not_ready_returns_none():
    connector = _GPIOPIRConnector(pin=17)
    connector._ready = False
    result = await connector.read()
    assert result is None


@pytest.mark.asyncio
async def test_gpio_connector_read_exception_returns_none():
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.IN = 1
    mock_gpio.input.side_effect = Exception("GPIO error")
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._GPIO_LIB", mock_gpio),
    ):
        connector = _GPIOPIRConnector(pin=17)
        result = await connector.read()
    assert result is None


def test_zenoh_connector_no_zenoh_lib():
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", False),
        patch("inputs.plugins.pir_motion._zenoh", None),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    assert connector._session is None


def test_zenoh_connector_subscribe_fails():
    mock_zenoh = MagicMock()
    mock_zenoh.open.side_effect = Exception("connection failed")
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    assert connector._session is None


def test_zenoh_on_message_motion_1():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    sample = MagicMock()
    sample.payload.decode.return_value = "MOTION:1"
    connector._on_message(sample)
    assert connector._queue.get_nowait() is True


def test_zenoh_on_message_motion_0():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    sample = MagicMock()
    sample.payload.decode.return_value = "MOTION:0"
    connector._on_message(sample)
    assert connector._queue.get_nowait() is False


def test_zenoh_on_message_unknown_payload():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    sample = MagicMock()
    sample.payload.decode.return_value = "GARBAGE"
    connector._on_message(sample)
    assert connector._queue.empty()


def test_zenoh_on_message_exception():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    sample = MagicMock()
    sample.payload.decode.side_effect = Exception("decode error")
    connector._on_message(sample)
    assert connector._queue.empty()


@pytest.mark.asyncio
async def test_zenoh_read_no_session_returns_none():
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", False),
        patch("inputs.plugins.pir_motion._zenoh", None),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    result = await connector.read()
    assert result is None


@pytest.mark.asyncio
async def test_zenoh_read_empty_queue_returns_none():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    result = await connector.read()
    assert result is None


def test_zenoh_stop_calls_undeclare_and_close():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    subscriber = connector._subscriber
    session = connector._session
    connector.stop()
    if subscriber:
        subscriber.undeclare.assert_called_once()
    if session:
        session.close.assert_called_once()


def test_zenoh_stop_undeclare_exception():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    if connector._subscriber:
        connector._subscriber.undeclare = MagicMock(
            side_effect=Exception("undeclare error")
        )
    connector.stop()


def test_zenoh_stop_close_exception():
    mock_zenoh = MagicMock()
    with (
        patch("inputs.plugins.pir_motion._ZENOH_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._zenoh", mock_zenoh),
    ):
        connector = _ZenohPIRConnector("om/sensors/pir")
    if connector._session:
        connector._session.close = MagicMock(side_effect=Exception("close error"))
    connector.stop()


def test_serial_connector_stop_closes_open_port():
    with patch("inputs.plugins.pir_motion._serial.Serial") as mock_ser:
        mock_ser.return_value.is_open = True
        connector = _SerialPIRConnector("/dev/ttyUSB0", 9600, 1.0)
        connector.stop()
        mock_ser.return_value.close.assert_called_once()


def test_gpio_connector_stop_calls_cleanup():
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.IN = 1
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._GPIO_LIB", mock_gpio),
    ):
        connector = _GPIOPIRConnector(pin=17)
        connector.stop()
        mock_gpio.cleanup.assert_called_once_with(17)


def test_gpio_connector_stop_cleanup_exception():
    mock_gpio = MagicMock()
    mock_gpio.BCM = 11
    mock_gpio.IN = 1
    mock_gpio.cleanup.side_effect = Exception("cleanup error")
    with (
        patch("inputs.plugins.pir_motion._GPIO_AVAILABLE", True),
        patch("inputs.plugins.pir_motion._GPIO_LIB", mock_gpio),
    ):
        connector = _GPIOPIRConnector(pin=17)
        connector.stop()
