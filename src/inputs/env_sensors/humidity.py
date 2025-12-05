import time
import random
import logging

logger = logging.getLogger(__name__)

class HumiditySensor:
    def __init__(self, config=None):
        self.config = config or {}
        self.name = self.config.get("name", "Humidity_Sensor")
        self.running = False

    def start(self):
        self.running = True
        logger.info(f"{self.name} started.")

    def read(self):
        if not self.running: return None
        humidity = round(random.uniform(40.0, 60.0), 2)
        time.sleep(0.1)
        return {
            "source": "humidity",
            "payload": {"humidity_percent": humidity}
        }

    def stop(self):
        self.running = False
