import json
import logging
from typing import Optional

import serial as _pyserial
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.fall_recovery.interface import FallRecoveryAction, FallRecoveryInput
from providers.imu_provider import IMUProvider


class FallRecoverySerialConfig(ActionConfig):
    """
    Configuration for Fall Recovery Serial connector.

    Parameters
    ----------
    port : str
        Serial port for the robot controller.
    baudrate : int
        Serial communication baudrate.
    timeout : float
        Serial write timeout in seconds.
    """

    port: str = Field(
        default="/dev/ttyUSB0",
        description="Serial port for robot controller",
    )
    baudrate: int = Field(
        default=115200,
        description="Serial communication baudrate",
    )
    timeout: float = Field(
        default=2.0,
        description="Serial write timeout in seconds",
    )


class FallRecoverySerialConnector(
    ActionConnector[FallRecoverySerialConfig, FallRecoveryInput]
):
    """
    Serial connector for fall recovery actions.

    Sends recovery commands to robot controller via serial port.
    Compatible with Arduino-based controllers or any serial-capable
    robot platform.
    """

    def __init__(self, config: FallRecoverySerialConfig):
        """
        Initialize the FallRecoverySerialConnector.

        Parameters
        ----------
        config : FallRecoverySerialConfig
            Configuration for the connector.
        """
        super().__init__(config)

        self.ser: Optional[_pyserial.Serial] = None
        self.imu_provider = IMUProvider()

        try:
            self.ser = _pyserial.Serial(
                config.port, config.baudrate, timeout=config.timeout
            )
            logging.info(f"FallRecoverySerialConnector: connected to {config.port}")
        except Exception as e:
            logging.error(
                f"FallRecoverySerialConnector: failed to open serial port - {e}"
            )

    def _send_command(self, command: dict) -> bool:
        """
        Send a JSON command via serial port.

        Parameters
        ----------
        command : dict
            Command dictionary to serialize and send.

        Returns
        -------
        bool
            True if sent successfully, False otherwise.
        """
        if self.ser is None:
            logging.error("FallRecoverySerialConnector: serial port not available")
            return False

        try:
            payload = json.dumps(command) + "\n"
            self.ser.write(payload.encode("utf-8"))
            logging.info(f"FallRecoverySerialConnector: sent command={command}")
            return True
        except Exception as e:
            logging.error(f"FallRecoverySerialConnector: error sending command - {e}")
            return False

    async def connect(self, output_interface: FallRecoveryInput) -> None:
        """
        Execute a fall recovery action.

        Parameters
        ----------
        output_interface : FallRecoveryInput
            Input containing the recovery action to perform.
        """
        action = output_interface.action
        message = output_interface.message

        logging.info(
            f"FallRecoverySerialConnector: executing action={action.value} "
            f"message='{message}'"
        )

        if action == FallRecoveryAction.STAND_UP:
            self._send_command({"action": "stand_up", "message": message})
            self.imu_provider.reset_alerts()

        elif action == FallRecoveryAction.EMERGENCY_STOP:
            self._send_command({"action": "emergency_stop", "message": message})

        elif action == FallRecoveryAction.ALERT_OPERATOR:
            logging.warning(f"FallRecoverySerialConnector: operator alert - {message}")
            self._send_command({"action": "alert_operator", "message": message})

        else:
            logging.warning(
                f"FallRecoverySerialConnector: unknown action '{action.value}'"
            )
