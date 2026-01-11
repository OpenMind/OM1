import asyncio
import json
import logging
import time
import typing as T

import psutil

from ..base import Sensor, SensorConfig

# Setup logger for the plugin
logger = logging.getLogger(__name__)


class SystemMonitorSensor(Sensor[dict]):
    """
    SystemMonitorSensor: Monitors hardware health (CPU, RAM, Temperature).
    Inherits from Sensor[R] where R is a dictionary of metrics.
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.name = "system_monitor"
        # Extract mock_mode from config (loaded via kwargs in SensorConfig)
        self.mock_mode = getattr(config, "mock_mode", False)
        self.latest_data: dict = {}
        logger.info(f"SystemMonitorSensor initialized (Mock: {self.mock_mode})")

    async def _listen_loop(self) -> T.AsyncIterator[dict]:
        """
        Main loop: continuously yields system metrics at set intervals.
        This is the required implementation for the Sensor base class listen() method.
        """
        while True:
            try:
                data = self._collect_metrics()
                self.latest_data = data  # Cache data for formatted_latest_buffer
                yield data

                # Sleep for 2 seconds to avoid overloading the CPU
                await asyncio.sleep(2.0)

            except Exception as e:
                logger.error(f"Error in system monitor loop: {e}")
                await asyncio.sleep(5.0)  # Sleep longer on error before retrying

    def formatted_latest_buffer(self) -> str | None:
        """
        Returns a formatted string description of the current system status
        for the Fuser (AI Agent) to read.
        """
        if not self.latest_data:
            return None

        # Format as a readable string for the LLM
        return (
            f"System Status -> "
            f"CPU: {self.latest_data.get('cpu_usage_percent', 0)}% | "
            f"RAM: {self.latest_data.get('memory_usage_percent', 0)}% | "
            f"Temp: {self.latest_data.get('temperature_c', 'N/A')}C"
        )

    async def raw_to_text(self, raw_input: dict) -> str:
        """
        Converts raw input data (dict) into text format (JSON) for logging or processing.
        """
        return json.dumps(raw_input)

    def _collect_metrics(self) -> dict:
        """
        Helper method: Gathers actual hardware metrics using psutil.
        """
        return {
            "cpu_usage_percent": psutil.cpu_percent(interval=None),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "temperature_c": self._get_temp(),
            "timestamp": time.time(),
        }

    def _get_temp(self):
        """
        Safely retrieves CPU temperature.
        """
        if self.mock_mode:
            return 45.0

        try:
            temps = psutil.sensors_temperatures()
            # Check for common thermal sensor names across different platforms
            for name in ["cpu_thermal", "coretemp", "soc_thermal", "k10temp"]:
                if name in temps:
                    return temps[name][0].current
            return "N/A"
        except (AttributeError, KeyError, PermissionError):
            return "N/A"
