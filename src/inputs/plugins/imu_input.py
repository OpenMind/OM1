import asyncio
import json
import logging
import time
from typing import Optional

import serial
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.imu_provider import IMUProvider
from providers.io_provider import IOProvider


class IMUConfig(SensorConfig):
    """Configuration for the IMU input plugin."""

    port: str = Field(
        default="/dev/ttyUSB0",
        description="Serial port for IMU device (e.g., /dev/ttyUSB0 or COM3)",
    )
    baudrate: int = Field(
        default=115200,
        description="Serial communication baudrate",
    )
    timeout: float = Field(
        default=1.0,
        description="Read timeout in seconds",
    )
    fall_threshold: float = Field(
        default=45.0,
        description="Roll/pitch angle threshold in degrees to detect a fall",
    )
    impact_threshold: float = Field(
        default=20.0,
        description="Acceleration magnitude threshold (m/s^2) to detect an impact",
    )
    poll_interval: float = Field(
        default=0.1,
        description="Polling interval in seconds",
    )


class IMUInput(FuserInput[IMUConfig, Optional[dict]]):
    """
    IMU sensor input plugin for OM1.

    Reads accelerometer, gyroscope, and orientation data from an IMU
    sensor connected via serial port (e.g., MPU6050 or BNO055 with
    Arduino/serial bridge). Updates IMUProvider with latest readings
    and provides fall/impact context to the LLM.

    Expected serial data format (JSON per line):
        {"ax": 0.1, "ay": 0.2, "az": 9.8,
         "gx": 0.0, "gy": 0.0, "gz": 0.0,
         "roll": 1.2, "pitch": 0.5, "yaw": 90.0}
    """

    def __init__(self, config: IMUConfig):
        """
        Initialize the IMU input plugin.

        Parameters
        ----------
        config : IMUConfig
            Configuration for the IMU sensor.
        """
        super().__init__(config)

        self.ser = None
        self.io_provider = IOProvider()
        self.imu_provider = IMUProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "IMU Sensor (Accelerometer, Gyroscope, Orientation)"

        # Apply thresholds to provider
        self.imu_provider.fall_threshold = config.fall_threshold
        self.imu_provider.impact_threshold = config.impact_threshold

        try:
            self.ser = serial.Serial(
                config.port, config.baudrate, timeout=config.timeout
            )
            logging.info(
                f"IMUInput: connected to {config.port} at {config.baudrate} baud"
            )
        except serial.SerialException as e:
            logging.error(f"IMUInput: failed to open serial port - {e}")

    async def _poll(self) -> Optional[dict]:
        """
        Poll IMU sensor for latest data.

        Returns
        -------
        Optional[dict]
            Parsed IMU data dictionary or None if unavailable.
        """
        await asyncio.sleep(self.config.poll_interval)

        if self.ser is None:
            return None

        try:
            line = self.ser.readline().decode("utf-8").strip()
            if not line:
                return None

            data = json.loads(line)
            logging.debug(f"IMUInput: raw data={data}")
            return data

        except Exception as e:
            logging.error(f"IMUInput: error reading data - {e}")
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """
        Convert raw IMU data to human-readable message for LLM.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw IMU data dictionary.

        Returns
        -------
        Optional[Message]
            Timestamped message or None.
        """
        if raw_input is None:
            return None

        try:
            ax = float(raw_input.get("ax", 0.0))
            ay = float(raw_input.get("ay", 0.0))
            az = float(raw_input.get("az", 0.0))
            gx = float(raw_input.get("gx", 0.0))
            gy = float(raw_input.get("gy", 0.0))
            gz = float(raw_input.get("gz", 0.0))
            roll = float(raw_input.get("roll", 0.0))
            pitch = float(raw_input.get("pitch", 0.0))
            yaw = float(raw_input.get("yaw", 0.0))

            self.imu_provider.update(ax, ay, az, gx, gy, gz, roll, pitch, yaw)

            state = self.imu_provider.state

            if state["is_fallen"]:
                message = (
                    f"WARNING: Robot has fallen! "
                    f"Roll={roll:.1f}deg, Pitch={pitch:.1f}deg. "
                    f"Immediate recovery action required."
                )
            elif state["impact_detected"]:
                accel_mag = (ax**2 + ay**2 + az**2) ** 0.5
                message = (
                    f"WARNING: Impact detected! "
                    f"Acceleration magnitude={accel_mag:.2f} m/s^2. "
                    f"Check robot integrity."
                )
            else:
                message = (
                    f"IMU status normal. "
                    f"Orientation: roll={roll:.1f}deg, pitch={pitch:.1f}deg, "
                    f"yaw={yaw:.1f}deg."
                )

            return Message(timestamp=time.time(), message=message)

        except Exception as e:
            logging.error(f"IMUInput: error processing data - {e}")
            return None

    async def raw_to_text(self, raw_input: Optional[dict]):
        """
        Update message buffer with processed IMU data.

        Parameters
        ----------
        raw_input : Optional[dict]
            Raw IMU data to process.
        """
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string for LLM context, or None if buffer is empty.
        """
        if not self.messages:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
