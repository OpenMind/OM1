import time, yaml, random, threading
class MQ135Plugin:
    def __init__(self, cfg="src/plugins/sensors/config.yaml"):
        with open(cfg) as f: c = yaml.safe_load(f)
        self.mock = c.get("mock_mode", True)
        self.data = {"co2_ppm":450,"air_quality":"good"}
        threading.Thread(target=self._run,daemon=True).start(); time.sleep(0.5)
    def _run(self):
        while True:
            if self.mock:
                p = random.randint(300,1200)
                q = "good" if p<600 else "moderate" if p<900 else "poor"
                self.data = {"co2_ppm":p,"air_quality":q}
            time.sleep(3)
    def get_data(self): return self.data.copy()
