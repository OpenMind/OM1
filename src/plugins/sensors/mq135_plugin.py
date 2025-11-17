import time, yaml, random, threading

class MQ135Plugin:
    def __init__(self, config_path="src/plugins/sensors/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.mock = cfg.get("mock_mode", True)
        self.data = {"co2_ppm": 450, "air_quality": "good"}
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        while True:
            if self.mock:
                ppm = random.randint(300, 1200)
                quality = "good" if ppm < 600 else "moderate" if ppm < 900 else "poor"
                self.data = {"co2_ppm": ppm, "air_quality": quality}
            time.sleep(3)

    def get_data(self): return self.data.copy()
