"""
Minimal Teleops Connection Background
通过 HTTP API 定期报告设备状态到 Portal，让设备显示在线
"""

import logging
import threading
import time
import requests

from backgrounds.base import Background, BackgroundConfig


class TeleopsConnection(Background):
    """
    Minimal HTTP-based status reporter to OpenMind Teleops Portal.
    Sends status updates every 5 seconds to keep device showing as online.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        self.api_key = getattr(config, "api_key", None)
        if not self.api_key:
            logging.warning("No API key provided for Teleops connection")
            return

        self.status_url = "https://api.openmind.org/api/core/teleops/status"
        self._running = True

        # Start status update thread
        self._status_thread = threading.Thread(
            target=self._status_loop, daemon=True
        )
        self._status_thread.start()

        logging.info("✅ TeleopsConnection: Started status reporter - Device should appear ONLINE")

    def _status_loop(self):
        """
        Send status update every 5 seconds to keep device showing as ONLINE.
        Uses the correct TeleopsStatus format as defined in TeleopsStatusProvider.
        """
        while self._running:
            try:
                # Create properly formatted TeleopsStatus
                # Matching the format in src/providers/teleops_status_provider.py
                current_time = str(time.time())
                status_message = {
                    "machine_name": "Spot",
                    "update_time": current_time,
                    "battery_status": {
                        "battery_level": 100.0,
                        "temperature": 25.0,
                        "voltage": 12.0,
                        "timestamp": current_time,
                        "charging_status": False
                    },
                    "action_status": {
                        "action": "AI",  # Must be "action" not "action_type"
                        "timestamp": current_time
                    },
                    "video_connected": True
                }

                # Send via HTTP POST (not WebSocket)
                response = requests.post(
                    self.status_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=status_message
                )

                if response.status_code == 200:
                    logging.info("✓ Teleops status update sent (200 OK) - Device should appear ONLINE")
                else:
                    logging.warning(f"Status update failed: {response.status_code} - {response.text}")

            except Exception as e:
                logging.error(f"Teleops status update error: {e}")

            time.sleep(5)
