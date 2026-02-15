import logging
import time
from typing import List, Optional

from inputs.base import Message
from inputs.base.loop import FuserInput
from inputs.plugins.unitree_go2_battery import (
    UnitreeGo2Battery,
    UnitreeGo2BatteryConfig,
)
from providers.io_provider import IOProvider
from tests.integration.mock_inputs.data_providers.mock_state_provider import (
    get_next_battery,
)


class MockUnitreeGo2Battery(UnitreeGo2Battery):
    """
    Mock implementation of UnitreeGo2Battery that uses the central state provider.

    This class bypasses the Unitree SDK and hardware to inject
    mock battery data for integration testing.
    """

    def __init__(self, config: UnitreeGo2BatteryConfig = UnitreeGo2BatteryConfig()):
        """
        Initialize with mock state provider, bypassing real hardware.

        Parameters
        ----------
        config : UnitreeGo2BatteryConfig, optional
            Configuration for the sensor
        """
        # Skip UnitreeGo2Battery.__init__ to avoid Unitree SDK setup
        FuserInput.__init__(self, config)

        self.io_provider = IOProvider()
        self.messages: list[Message] = []
        self.descriptor_for_LLM = "MOCK Energy Levels (Integration Test)"

        self.battery_percentage = 0.0
        self.battery_voltage = 0.0
        self.battery_amperes = 0.0

        self.running = True
        self.data_processed = False

        logging.info("MockUnitreeGo2Battery initialized - using mock state provider")

    async def _poll(self) -> List[float]:
        """
        Poll for mock battery data from the state provider.

        Returns
        -------
        List[float]
            Battery data [percent, voltage, amperes]
        """
        data = get_next_battery()
        if data is not None:
            self.battery_percentage = data[0]
            self.battery_voltage = data[1]
            self.battery_amperes = data[2]
            logging.info(
                f"MockUnitreeGo2Battery: Battery {data[0]}%, {data[1]}V, {data[2]}A"
            )
            return data

        if not self.data_processed:
            logging.info("MockUnitreeGo2Battery: No more battery data to process")
            self.data_processed = True

        return [self.battery_percentage, self.battery_voltage, self.battery_amperes]

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[Message]:
        """Process raw battery data to generate text description."""
        battery_percentage = raw_input[0]

        if battery_percentage < 7:
            message = "CRITICAL: Your battery is almost empty. Immediately move to your charging station and recharge. If you cannot find your charging station, consider sitting down."
            return Message(timestamp=time.time(), message=message)
        elif battery_percentage < 15:
            message = "WARNING: You are low on energy. Move to your charging station and recharge."
            return Message(timestamp=time.time(), message=message)

        return None

    async def raw_to_text(self, raw_input: List[float]):
        """Convert raw battery data to text and update message buffer."""
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
        """Stop the mock battery input."""
        self.running = False
        logging.info("MockUnitreeGo2Battery: Stopped")

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        logging.info("MockUnitreeGo2Battery: Cleanup completed")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.cleanup()
