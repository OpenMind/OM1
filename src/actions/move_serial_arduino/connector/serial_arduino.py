import logging
import serial
from typing import Optional
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
# Import the interface from the local module to ensure correct action mapping
from actions.move_serial_arduino.interface import MoveInput


class MoveSerialConfig(ActionConfig):
    """Configuration schema for Arduino Serial connection."""
    
    port: str = Field(
        default="",
        description="Serial port (e.g., COM3, /dev/ttyUSB0). Leave empty for simulation mode.",
    )
    baudrate: int = Field(
        default=9600,
        description="Serial communication speed (baud rate).",
    )
    timeout: float = Field(
        default=1.0,
        description="Read/write timeout in seconds.",
    )


class MoveSerialConnector(ActionConnector[MoveSerialConfig, MoveInput]):
    """
    Serial connector for Arduino-based robot locomotion.
    Handles 'actuator:X' protocol sending over USB/UART.
    """

    # Static mapping for O(1) lookup performance
    CMD_MAP = {
        "be still": "0",
        "stop": "0",
        "small jump": "1",
        "medium jump": "2",
        "big jump": "3",
        "walk": "W",
        "walk forward": "W",
        "walk back": "S",
        "turn left": "A",
        "turn right": "D",
        "dance": "9"
    }

    def __init__(self, config: MoveSerialConfig):
        super().__init__(config)
        self.ser: Optional[serial.Serial] = None
        self.logger = logging.getLogger(__name__)

        if self.config.port:
            try:
                self.ser = serial.Serial(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.timeout
                )
                self.logger.info(f"[Arduino] Connected on {self.config.port} @ {self.config.baudrate} baud")
            except serial.SerialException as e:
                self.logger.error(f"[Arduino] Connection failed on {self.config.port}: {e}")
                self.ser = None

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Translates high-level actions to serial byte commands.
        """
        # Handle both Enum objects and raw strings for robustness
        raw_action = output_interface.action
        action_key = raw_action.value if hasattr(raw_action, 'value') else str(raw_action)
        
        cmd_char = self.CMD_MAP.get(action_key)

        if not cmd_char:
            self.logger.debug(f"[Arduino] Action '{action_key}' ignored (no mapping found).")
            return

        # Protocol format: actuator:{char}\r\n
        payload = f"actuator:{cmd_char}\r\n"
        
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(payload.encode("utf-8"))
                self.logger.info(f"[Arduino] Sent: {payload.strip()}")
            except Exception as e:
                self.logger.error(f"[Arduino] Write error: {e}")
        else:
            # Fallback for simulation or debug
            self.logger.warning(f"[Sim] Serial unavailable. Would send: {payload.strip()}")

    def tick(self) -> None:
        # TODO: Implement heartbeat or battery level check
        self.sleep(0.1)