import asyncio
import json
import logging
import time
import typing as T

import psutil

from ..base import Sensor, SensorConfig

# Setup logger
logger = logging.getLogger(__name__)


class DiskMonitorSensor(Sensor[dict]):
    """
    DiskMonitorSensor: Monitors disk usage statistics (Total, Used, Free, Percent).
    Crucial for preventing system crashes due to full storage (logs, models, docker).
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self.name = "disk_monitor"
        # Configurable path to monitor, default is root "/"
        self.mount_point = getattr(config, "mount_point", "/")
        self.latest_data: dict = {}
        logger.info(f"DiskMonitorSensor initialized monitoring: {self.mount_point}")

    async def _listen_loop(self) -> T.AsyncIterator[dict]:
        """
        Periodically yields disk usage statistics.
        Interval is set longer (10s) as disk usage changes slower than CPU/RAM.
        """
        while True:
            try:
                data = self._collect_disk_usage()
                self.latest_data = data
                yield data

                # Check disk usage every 10 seconds (resource efficient)
                await asyncio.sleep(10.0)

            except Exception as e:
                logger.error(f"Error in disk monitor loop: {e}")
                await asyncio.sleep(10.0)

    def formatted_latest_buffer(self) -> str | None:
        """
        Returns a readable summary for the AI Agent.
        Alerts if disk usage is critically high (>90%).
        """
        if not self.latest_data:
            return None

        percent = self.latest_data.get("percent", 0)
        status = "CRITICAL" if percent > 90 else "OK"

        return (
            f"Disk Status ({self.mount_point}) -> "
            f"Usage: {percent}% | "
            f"Free: {self.latest_data.get('free_gb', 0)} GB | "
            f"Health: {status}"
        )

    async def raw_to_text(self, raw_input: dict) -> str:
        """
        Converts metrics to JSON string.
        """
        return json.dumps(raw_input)

    def _collect_disk_usage(self) -> dict:
        """
        Uses psutil to fetch disk statistics safely.
        """
        try:
            usage = psutil.disk_usage(self.mount_point)
            return {
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent": usage.percent,
                "path": self.mount_point,
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.warning(f"Failed to get disk usage for {self.mount_point}: {e}")
            return {"error": str(e)}
		
