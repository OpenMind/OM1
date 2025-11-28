# src/plugins/sensors/air_quality_plugin.py
import time
import random
import threading


class AirQualityPlugin:
    """
    Simulated air-quality / CO₂ sensor.

    Produces:
      {
        "co2_ppm": <int>,
        "air_quality": "good" | "moderate" | "poor" | "very poor"
      }
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.data = {"co2_ppm": 500, "air_quality": "good"}
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        time.sleep(0.2)

    def _run(self):
        while True:
            co2 = random.randint(350, 2500)

            if co2 < 800:
                quality = "good"
            elif co2 < 1200:
                quality = "moderate"
            elif co2 < 1800:
                quality = "poor"
            else:
                quality = "very poor"

            self.data = {
                "co2_ppm": co2,
                "air_quality": quality,
            }
            time.sleep(self.interval)

    def get_data(self) -> dict:
        return dict(self.data)
