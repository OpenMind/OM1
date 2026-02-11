from unittest.mock import Mock, patch

import pytest

from actions.move.interface import MoveInput
from actions.move_serial_arduino.connector.serial_arduino import (
    MoveSerialConfig,
    MoveSerialConnector,
)


class TestMoveSerialConfig:
    """Test MoveSerialConfig configuration."""

    def test_default_config(self):
        """Test default configuration has empty port."""
        config = MoveSerialConfig()
        assert config.port == ""

    def test_custom_port(self):
        """Test custom port configuration."""
        config = MoveSerialConfig(port="/dev/cu.usbmodem14101")
        assert config.port == "/dev/cu.usbmodem14101"


class TestMoveSerialConnectorInit:
    """Test MoveSerialConnector initialization."""

    def test_init_without_port(self):
        """Test initialization without port (simulation mode)."""
        config = MoveSerialConfig()
        connector = MoveSerialConnector(config)
        assert connector.port == ""
        assert connector.ser is None

    def test_init_with_port(self):
        """Test initialization with port creates serial connection."""
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.serial.Serial"
        ) as mock_serial:
            mock_serial_instance = Mock()
            mock_serial.return_value = mock_serial_instance

            config = MoveSerialConfig(port="/dev/ttyUSB0")
            connector = MoveSerialConnector(config)

            assert connector.port == "/dev/ttyUSB0"
            mock_serial.assert_called_once_with("/dev/ttyUSB0", 9600)
            assert connector.ser == mock_serial_instance


class TestMoveSerialConnectorConnect:
    """Test connect method for each movement action."""

    @pytest.fixture
    def connector_no_serial(self):
        """Create connector without serial port (simulation mode)."""
        config = MoveSerialConfig()
        return MoveSerialConnector(config)

    @pytest.fixture
    def connector_with_serial(self):
        """Create connector with mocked serial port."""
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.serial.Serial"
        ) as mock_serial:
            mock_serial_instance = Mock()
            mock_serial_instance.is_open = True
            mock_serial.return_value = mock_serial_instance

            config = MoveSerialConfig(port="/dev/ttyUSB0")
            connector = MoveSerialConnector(config)
            return connector

    @pytest.mark.asyncio
    async def test_connect_be_still_simulation(self, connector_no_serial):
        """Test be still action in simulation mode."""
        move_input = MoveInput(action="be still")  # type: ignore[arg-type]
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.logging"
        ) as mock_logging:
            await connector_no_serial.connect(move_input)
            mock_logging.info.assert_called_with(
                "SerialNotOpen - Simulating transmit: actuator:0\r\n"
            )

    @pytest.mark.asyncio
    async def test_connect_small_jump_simulation(self, connector_no_serial):
        """Test small jump action in simulation mode."""
        move_input = MoveInput(action="small jump")  # type: ignore[arg-type]
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.logging"
        ) as mock_logging:
            await connector_no_serial.connect(move_input)
            mock_logging.info.assert_called_with(
                "SerialNotOpen - Simulating transmit: actuator:1\r\n"
            )

    @pytest.mark.asyncio
    async def test_connect_medium_jump_simulation(self, connector_no_serial):
        """Test medium jump action in simulation mode."""
        move_input = MoveInput(action="medium jump")  # type: ignore[arg-type]
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.logging"
        ) as mock_logging:
            await connector_no_serial.connect(move_input)
            mock_logging.info.assert_called_with(
                "SerialNotOpen - Simulating transmit: actuator:2\r\n"
            )

    @pytest.mark.asyncio
    async def test_connect_big_jump_simulation(self, connector_no_serial):
        """Test big jump action in simulation mode."""
        move_input = MoveInput(action="big jump")  # type: ignore[arg-type]
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.logging"
        ) as mock_logging:
            await connector_no_serial.connect(move_input)
            mock_logging.info.assert_called_with(
                "SerialNotOpen - Simulating transmit: actuator:3\r\n"
            )

    @pytest.mark.asyncio
    async def test_connect_be_still_with_serial(self, connector_with_serial):
        """Test be still action with actual serial connection."""
        move_input = MoveInput(action="be still")  # type: ignore[arg-type]
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.logging"
        ) as mock_logging:
            await connector_with_serial.connect(move_input)
            mock_logging.info.assert_called_with("SendToArduinoSerial: actuator:0\r\n")
            connector_with_serial.ser.write.assert_called_once_with(b"actuator:0\r\n")

    @pytest.mark.asyncio
    async def test_connect_small_jump_with_serial(self, connector_with_serial):
        """Test small jump action writes correct bytes to serial."""
        move_input = MoveInput(action="small jump")  # type: ignore[arg-type]
        with patch("actions.move_serial_arduino.connector.serial_arduino.logging"):
            await connector_with_serial.connect(move_input)
            connector_with_serial.ser.write.assert_called_once_with(b"actuator:1\r\n")

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector_no_serial):
        """Test unknown action logs info and simulates."""
        move_input = MoveInput(action="unknown")  # type: ignore[arg-type]
        with patch(
            "actions.move_serial_arduino.connector.serial_arduino.logging"
        ) as mock_logging:
            await connector_no_serial.connect(move_input)
            mock_logging.info.assert_any_call("Other move type: unknown")


class TestMoveSerialConnectorTick:
    """Test tick method."""

    def test_tick_calls_sleep(self):
        """Test tick calls sleep with 0.1 seconds."""
        config = MoveSerialConfig()
        connector = MoveSerialConnector(config)
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(0.1)
