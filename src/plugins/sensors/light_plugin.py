# src/plugins/sensors/light_plugin.py
import time
import random
import threading


class LightSensorPlugin:
    """
    Simulated ambient light sensor.

    Produces:
      {
        "lux": <float>,
        "description": "dark" | "dim" | "moderate" | "bright" | "very bright"
      }
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.data = {"lux": 200.0, "description": "moderate"}
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        time.sleep(0.2)

    def _run(self):
        while True:
            lux = random.uniform(0, 1200)

            if lux < 50:
                desc = "dark"
            elif lux < 200:
                desc = "dim"
            elif lux < 500:
                desc = "moderate"
            elif lux < 900:
                desc = "bright"
            else:
                desc = "very bright"

            self.data = {
                "lux": round(lux, 1),
                "description": desc,
            }
            time.sleep(self.interval)

    def get_data(self) -> dict:
        return dict(self.data)
