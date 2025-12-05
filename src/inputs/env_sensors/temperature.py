import time
import random
import logging

logger = logging.getLogger(__name__)

class TemperatureSensor:
    def __init__(self, config=None):
        self.config = config or {}
        self.name = self.config.get("name", "Temperature_Sensor")
        self.running = False

    def start(self):
        self.running = True
        logger.info(f"{self.name} started.")

    def read(self):
        if not self.running: return None
        # Mocking DHT22/BME280 behavior
        temp = round(random.uniform(20.0, 30.0), 2)
        time.sleep(0.1)
        return {
            "source": "temperature",
            "payload": {"temperature_c": temp, "unit": "celsius"}
        }

    def stop(self):
        self.running = False
