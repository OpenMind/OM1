import time, yaml, random, threading
class DHT22Plugin:
    def __init__(self, cfg="src/plugins/sensors/config.yaml"):
        with open(cfg) as f: c = yaml.safe_load(f)
        self.mock = c.get("mock_mode", True)
        self.data = {"temperature":22.0,"humidity":55.0,"comfort":"comfortable"}
        threading.Thread(target=self._run,daemon=True).start(); time.sleep(0.5)
    def _run(self):
        while True:
            if self.mock:
                t = round(random.uniform(20,26),1); h = round(random.uniform(40,65),1)
                c = "comfortable" if 22<=t<=25 else "uncomfortable"
                self.data = {"temperature":t,"humidity":h,"comfort":c}
            time.sleep(2)
    def get_data(self): return self.data.copy()
