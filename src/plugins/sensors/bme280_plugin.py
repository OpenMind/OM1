import time, yaml, random, threading
from typing import Dict

class BME280Plugin:
    def __init__(self, config_path="src/plugins/sensors/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.mock = cfg.get("mock_mode", True)
        self.data = {"temperature": 20.0, "humidity": 50.0, "pressure": 1013.0, "comfort": "comfortable"}
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        while True:
            if self.mock:
                temp = round(random.uniform(18, 28), 2)
                hum = round(random.uniform(35, 75), 2)
                press = round(random.uniform(990, 1030), 2)
                comfort = "cold" if temp < 18 else "hot" if temp > 26 else "dry" if hum < 30 else "humid" if hum > 70 else "comfortable"
                self.data = {"temperature": temp, "humidity": hum, "pressure": press, "comfort": comfort}
            time.sleep(1)

    def get_data(self) -> Dict: return self.data.copy()
