import logging
import os
import threading
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers import BatteryStatus, TeleopsStatus, TeleopsStatusProvider
from providers.config_provider import ConfigProvider


class DesktopStatusConfig(BackgroundConfig):
    """
    Configuration for Desktop Status Background.
    Sends periodic status updates to dashboard for desktop devices.
    """

    machine_name: str = Field(default="desktop_device", description="Machine name for dashboard")
    update_interval: int = Field(default=10, description="Status update interval in seconds")


class DesktopStatus(Background[DesktopStatusConfig]):
    """
    Desktop Status Background Plugin.
    Sends periodic teleops status updates so device appears on dashboard.
    Works on any platform (Mac, Linux, Windows).
    """

    def __init__(self, config: DesktopStatusConfig):
        super().__init__(config)

        # Get API key from global config or environment
        try:
            config_provider = ConfigProvider()
            api_key = config_provider.get("api_key") or os.environ.get("OM_API_KEY")
        except:
            api_key = os.environ.get("OM_API_KEY")

        if not api_key:
            logging.error("API key not found for DesktopStatus")
            return

        self.machine_name = self.config.machine_name
        self.update_interval = self.config.update_interval
        self.status_provider = TeleopsStatusProvider(api_key=api_key)
        self.running = True

        # Start background thread
        self.thread = threading.Thread(target=self._status_loop, daemon=True)
        self.thread.start()
        logging.info(f"DesktopStatus started - sending status every {self.update_interval}s")

    def _status_loop(self):
        """Background loop to send status updates."""
        while self.running:
            try:
                status = TeleopsStatus(
                    machine_name=self.machine_name,
                    update_time=str(time.time()),
                    battery_status=BatteryStatus(
                        battery_level=100.0,
                        temperature=25.0,
                        voltage=12.0,
                        timestamp=str(time.time()),
                        charging_status=True
                    ),
                    video_connected=True
                )
                self.status_provider.share_status(status)
                logging.debug(f"DesktopStatus: sent status update for {self.machine_name}")
            except Exception as e:
                logging.error(f"DesktopStatus error: {e}")

            time.sleep(self.update_interval)

    def stop(self):
        """Stop the status loop."""
        self.running = False
        logging.info("DesktopStatus stopped")
