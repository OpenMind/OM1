import logging
import time
from queue import Queue
from typing import List, Optional

from inputs.base import Message
from inputs.base.loop import FuserInput
from inputs.plugins.unitree_go2_odom import UnitreeGo2Odom, UnitreeGo2OdomConfig
from providers.io_provider import IOProvider
from providers.odom_provider_base import RobotState
from tests.integration.mock_inputs.data_providers.mock_state_provider import (
    get_next_odometry,
)


class MockUnitreeGo2Odom(UnitreeGo2Odom):
    """
    Mock implementation of UnitreeGo2Odom that uses the central state provider.

    This class bypasses the UnitreeGo2OdomProvider and hardware to inject
    mock odometry data for integration testing.
    """

    def __init__(self, config: UnitreeGo2OdomConfig = UnitreeGo2OdomConfig()):
        """
        Initialize with mock state provider, bypassing real hardware.

        Parameters
        ----------
        config : UnitreeGo2OdomConfig, optional
            Configuration for the sensor
        """
        # Skip UnitreeGo2Odom.__init__ to avoid UnitreeGo2OdomProvider setup
        FuserInput.__init__(self, config)

        self.io_provider = IOProvider()
        self.messages: List[Message] = []
        self.message_buffer: Queue[str] = Queue()
        self.descriptor_for_LLM = "MOCK Odometry (Integration Test)"

        self.running = True
        self.data_processed = False

        logging.info("MockUnitreeGo2Odom initialized - using mock state provider")

    async def _poll(self) -> Optional[dict]:
        """
        Poll for mock odometry data from the state provider.

        Returns
        -------
        Optional[dict]
            Odometry data dict or None
        """
        data = get_next_odometry()
        if data is not None:
            logging.info(
                f"MockUnitreeGo2Odom: Position x={data.get('x')}, y={data.get('y')}, "
                f"moving={data.get('moving')}"
            )
            # Convert body_attitude string to RobotState enum if present
            if "body_attitude" in data and isinstance(data["body_attitude"], str):
                attitude_str = data["body_attitude"].upper()
                if attitude_str == "SITTING":
                    data["body_attitude"] = RobotState.SITTING
                else:
                    data["body_attitude"] = RobotState.STANDING
            return data

        if not self.data_processed:
            logging.info("MockUnitreeGo2Odom: No more odometry data to process")
            self.data_processed = True

        return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        """Process raw odometry data to generate a timestamped message."""
        if raw_input is None:
            return None

        res = ""
        moving = raw_input.get("moving", False)
        attitude = raw_input.get("body_attitude", RobotState.STANDING)

        if attitude is RobotState.SITTING:
            res = "You are sitting down - do not generate new movement commands. "
        elif moving:
            res = "You are moving - do not generate new movement commands. "
        else:
            res = "You are standing still - you can move if you want to. "

        return Message(timestamp=time.time(), message=res)

    async def raw_to_text(self, raw_input: Optional[dict]):
        """Convert raw odometry data to text and update message buffer."""
        if raw_input is None:
            return

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
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result

    def stop(self):
        """Stop the mock odometry input."""
        self.running = False
        logging.info("MockUnitreeGo2Odom: Stopped")

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        logging.info("MockUnitreeGo2Odom: Cleanup completed")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()
