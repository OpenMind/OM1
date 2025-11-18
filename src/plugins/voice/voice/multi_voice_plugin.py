import time
import yaml
import random
import threading
from typing import List, Dict

class MultiVoicePlugin:
    def __init__(self, config_path="src/plugins/voice/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.mock = cfg.get("mock_mode", True)
        self.num_mics = cfg.get("num_mics", 4)
        self.data = {"speakers": [], "beamformed_text": "", "dominant_angle": 0}
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)

    def _run(self):
        while True:
            if self.mock:
                num_speakers = random.randint(1, 3)
                speakers = []
                texts = []
                angle = random.uniform(-60, 60)
                for i in range(num_speakers):
                    speaker = random.choice(["user1", "user2", "user3"])
                    text = random.choice([
                        "avanti", "stop", "gira a destra", "accendi luce", "ciao robot"
                    ])
                    speakers.append({"speaker": speaker, "text": text, "confidence": round(random.uniform(0.8, 0.99), 2)})
                    texts.append(text)
                self.data = {
                    "speakers": speakers,
                    "beamformed_text": " | ".join(texts),
                    "dominant_angle": round(angle, 1),
                    "active_speakers": num_speakers
                }
            time.sleep(1.8)

    def get_data(self):
        return self.data.copy()
