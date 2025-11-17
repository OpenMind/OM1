import random
import threading
import time

import yaml


class BME280Plugin:
    def __init__(self, cfg="src/plugins/sensors/config.yaml"):
        with open(cfg) as f:
            c = yaml.safe_load(f)
        self.mock = c.get("mock_mode", True)
        self.data = {
            "temperature": 20.0,
            "humidity": 50.0,
            "pressure": 1013.0,
            "comfort": "comfortable",
        }
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        while True:
            if self.mock:
                t = round(random.uniform(18, 28), 2)
                h = round(random.uniform(35, 75), 2)
                p = round(random.uniform(990, 1030), 2)
                c = (
                    "cold"
                    if t < 18
                    else (
                        "hot"
                        if t > 26
                        else "dry" if h < 30 else "humid" if h > 70 else "comfortable"
                    )
                )
                self.data = {"temperature": t, "humidity": h, "pressure": p, "comfort": c}
            time.sleep(1)

    def get_data(self):
        return self.data.copy()
