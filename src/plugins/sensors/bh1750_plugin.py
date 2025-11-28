import time, yaml, random, threading
class BH1750Plugin:
    def __init__(self, cfg="src/plugins/sensors/config.yaml"):
        with open(cfg) as f: c = yaml.safe_load(f)
        self.mock = c.get("mock_mode", True)
        self.data = {"lux":300,"description":"moderate"}
        threading.Thread(target=self._run,daemon=True).start(); time.sleep(0.5)
    def _run(self):
        while True:
            if self.mock:
                l = random.randint(50,1000)
                d = ["dark","dim","moderate","bright","very bright"][min(l//200,4)]
                self.data = {"lux":l,"description":d}
            time.sleep(1)
    def get_data(self): return self.data.copy()
