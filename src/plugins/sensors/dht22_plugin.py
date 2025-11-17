import time, yaml, random, threading


class DHT22Plugin:
    def __init__(self, config_path="src/plugins/sensors/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.mock = cfg.get("mock_mode", True)
        self.data = {"temperature": 22.0, "humidity": 55.0, "comfort": "comfortable"}
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        while True:
            if self.mock:
                temp = round(random.uniform(20, 26), 1)
                hum = round(random.uniform(40, 65), 1)
                comfort = "comfortable" if 22 <= temp <= 25 else "uncomfortable"
                self.data = {"temperature": temp, "humidity": hum, "comfort": comfort}
            time.sleep(2)

    def get_data(self):
        return self.data.copy()
