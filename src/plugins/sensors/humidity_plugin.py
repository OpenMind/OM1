# src/plugins/sensors/humidity_plugin.py
import time
import random
import threading


class HumidityPlugin:
    """
    Simulated temperature + humidity sensor.

    Produces:
      {
        "temperature": <float>,  # °C
        "humidity": <float>,     # %
        "comfort": "dry" | "comfortable" | "humid"
      }
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.data = {
            "temperature": 24.0,
            "humidity": 50.0,
            "comfort": "comfortable",
        }
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        time.sleep(0.2)

    def _run(self):
        while True:
            temp = random.uniform(18.0, 32.0)
            humidity = random.uniform(30.0, 80.0)

            if humidity < 35:
                comfort = "dry"
            elif humidity <= 65:
                comfort = "comfortable"
            else:
                comfort = "humid"

            self.data = {
                "temperature": round(temp, 2),
                "humidity": round(humidity, 1),
                "comfort": comfort,
            }
            time.sleep(self.interval)

    def get_data(self) -> dict:
        return dict(self.data)
