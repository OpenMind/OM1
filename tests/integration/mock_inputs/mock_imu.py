import asyncio
import logging
import time
from typing import Optional

from inputs.base import Message
from inputs.base.loop import FuserInput
from inputs.plugins.unitree_go2_imu import UnitreeGo2IMU, UnitreeGo2IMUConfig
from providers.io_provider import IOProvider
from tests.integration.mock_inputs.data_providers.mock_state_provider import (
    get_next_imu,
)


class MockUnitreeGo2IMU(UnitreeGo2IMU):
    """
    Mock implementation of UnitreeGo2IMU that uses the central state provider.

    This class bypasses the Unitree SDK and CycloneDDS hardware to inject
    mock IMU data for integration testing.
    """

    def __init__(self, config: UnitreeGo2IMUConfig = UnitreeGo2IMUConfig()):
        """
        Initialize with mock state provider, bypassing real hardware.

        Parameters
        ----------
        config : UnitreeGo2IMUConfig, optional
            Configuration for the sensor
        """
        # Skip UnitreeGo2IMU.__init__ to avoid Unitree SDK setup
        FuserInput.__init__(self, config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "MOCK Body Orientation (Integration Test)"

        self.roll_deg = 0.0
        self.pitch_deg = 0.0

        self.running = True
        self.data_processed = False

        logging.info("MockUnitreeGo2IMU initialized - using mock state provider")

    async def _poll(self) -> Optional[str]:
        """
        Poll for mock IMU data from the state provider and evaluate tilt status.

        Returns
        -------
        Optional[str]
            Alert message if tilt exceeds thresholds, None otherwise.
        """
        data = get_next_imu()
        if data is not None:
            self.roll_deg = data.get("roll_deg", 0.0)
            self.pitch_deg = data.get("pitch_deg", 0.0)
            logging.info(
                f"MockUnitreeGo2IMU: roll={self.roll_deg}, pitch={self.pitch_deg}"
            )
        elif not self.data_processed:
            logging.info("MockUnitreeGo2IMU: No more IMU data to process")
            self.data_processed = True

        await asyncio.sleep(0.1)

        abs_roll = abs(self.roll_deg)
        abs_pitch = abs(self.pitch_deg)
        fall_threshold = self.config.fall_threshold_deg
        warning_threshold = self.config.warning_threshold_deg

        if abs_roll > fall_threshold or abs_pitch > fall_threshold:
            return (
                f"CRITICAL: You have fallen over. "
                f"Roll: {self.roll_deg} degrees, Pitch: {self.pitch_deg} degrees. "
                f"Use the recover action to stand back up immediately."
            )

        if abs_roll > warning_threshold or abs_pitch > warning_threshold:
            return (
                f"WARNING: You are tilting dangerously. "
                f"Roll: {self.roll_deg} degrees, Pitch: {self.pitch_deg} degrees. "
                f"Try to stabilize yourself."
            )

        return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """Process raw tilt status string to a timestamped Message."""
        if raw_input is not None:
            return Message(timestamp=time.time(), message=raw_input)
        return None

    async def raw_to_text(self, raw_input: Optional[str]):
        """Convert raw IMU data to text and update message buffer."""
        pending_message = await self._raw_to_text(raw_input)
        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents."""
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result

    def stop(self):
        """Stop the mock IMU input."""
        self.running = False
        logging.info("MockUnitreeGo2IMU: Stopped")

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        logging.info("MockUnitreeGo2IMU: Cleanup completed")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()
