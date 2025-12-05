import time
import random
import logging

logger = logging.getLogger(__name__)

class AirQualitySensor:
    def __init__(self, config=None):
        self.config = config or {}
        self.name = self.config.get("name", "AirQuality_Sensor")
        self.running = False

    def start(self):
        self.running = True
        logger.info(f"{self.name} started.")

    def read(self):
        if not self.running: return None
        # Mocking MQ135 behavior
        co2 = int(random.uniform(400, 1200))
        quality = "poor" if co2 > 1000 else "good"
        time.sleep(0.1)
        return {
            "source": "air_quality",
            "payload": {"co2_ppm": co2, "quality": quality}
        }

    def stop(self):
        self.running = False
