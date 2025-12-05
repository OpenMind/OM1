import time
import random
import logging

logger = logging.getLogger(__name__)

class LightSensor:
    def __init__(self, config=None):
        self.config = config or {}
        self.name = self.config.get("name", "Light_Sensor")
        self.running = False

    def start(self):
        self.running = True
        logger.info(f"{self.name} started.")

    def read(self):
        if not self.running: return None
        # Mocking BH1750 behavior
        lux = int(random.uniform(200, 1000))
        desc = "bright" if lux > 500 else "dim"
        time.sleep(0.1)
        return {
            "source": "light",
            "payload": {"lux": lux, "description": desc}
        }

    def stop(self):
        self.running = False
