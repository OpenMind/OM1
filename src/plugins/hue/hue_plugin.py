import time, yaml, random, threading
from typing import Dict, Any

class HuePlugin:
    def __init__(self, config_path="src/plugins/hue/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.mock = cfg.get("mock_mode", True)
        self.bridge_ip = cfg.get("bridge_ip", "192.168.1.100")
        self.username = cfg.get("username", "your_hue_username")
        self.data = {"lights": {}, "groups": {}, "scenes": {}}
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        while True:
            if self.mock:
                lights = {}
                for i in range(1, random.randint(4, 6)):
                    lights[f"light_{i}"] = {
                        "name": f"Lampada {i}",
                        "on": random.choice([True, False]),
                        "bri": random.randint(50, 254),
                        "hue": random.randint(0, 65535),
                        "sat": random.randint(0, 254),
                        "ct": random.randint(153, 500) if random.random() > 0.5 else None,
                        "reachable": True
                    }
                self.data = {
                    "lights": lights,
                    "groups": {"1": "Living Room", "2": "Kitchen"},
                    "scenes": {"relax": "Warm white", "energize": "Cool bright"}
                }
            time.sleep(2)

    def get_data(self) -> Dict[str, Any]:
        return self.data.copy()

    def set_light(self, light_id: str, **kwargs) -> Dict[str, Any]:
        if self.mock:
            if light_id in self.data["lights"]:
                self.data["lights"][light_id].update(kwargs)
            return {"success": True}
        return {"success": False, "error": "hardware not implemented yet"}
