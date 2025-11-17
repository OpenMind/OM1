import time, yaml, random, threading

class BH1750Plugin:
    def __init__(self, config_path="src/plugins/sensors/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.mock = cfg.get("mock_mode", True)
        self.data = {"lux": 300, "description": "moderate"}
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        while True:
            if self.mock:
                lux = random.randint(50, 1000)
                desc = ["dark", "dim", "moderate", "bright", "very bright"][min(lux//200, 4)]
                self.data = {"lux": lux, "description": desc}
            time.sleep(1)

    def get_data(self): return self.data.copy()
