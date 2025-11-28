# src/plugins/sensors/temperature_plugin.py
import time
import random
import threading


class TemperaturePlugin:
    """
    Simple simulated temperature sensor.

    Produces:
      {
        "temperature": <float>,  # °C
        "comfort": "cold" | "comfortable" | "hot"
      }
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.data = {"temperature": 24.0, "comfort": "comfortable"}
        # background thread that updates readings
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        # small delay so first reading is ready when called
        time.sleep(0.2)

    def _run(self):
        while True:
            temp = random.uniform(18.0, 34.0)

            if temp < 20:
                comfort = "cold"
            elif temp < 28:
                comfort = "comfortable"
            else:
                comfort = "hot"

            self.data = {
                "temperature": round(temp, 2),
                "comfort": comfort,
            }
            time.sleep(self.interval)

    def get_data(self) -> dict:
        # return a copy so callers can’t mutate internal state
        return dict(self.data)
