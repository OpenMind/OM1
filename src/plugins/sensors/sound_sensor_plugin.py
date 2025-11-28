import time, random, threading

class SoundSensorPlugin:
    def __init__(self):
        self.data = {"sound_db": 35.0, "environment": "quiet"}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            # random dB between 25 – 100
            db = random.uniform(25, 100)
            if db < 40:
                env = "quiet"
            elif db < 65:
                env = "moderate"
            else:
                env = "noisy"

            self.data = {"sound_db": round(db, 1), "environment": env}
            time.sleep(1)

    def get_data(self):
        return self.data.copy()
