# Author: Wanbogang (@Wanbogang) — Contribution for Issue #365 (OM1 Bounty Program)
# License: MIT

import math, time

class BME280Driver:
    """Driver sederhana. Jika MOCK=1 → data sintetis; kalau tidak, coba real driver."""
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._t0 = time.time()
        self._real = False
        if not cfg.get("mock"):
            try:
                # Real hardware (opsional): butuh board, busio, adafruit_bme280
                import board, busio, adafruit_bme280
                i2c = busio.I2C(board.SCL, board.SDA)
                self._bme = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=cfg["i2c_addr"])
                self._real = True
            except Exception:
                # Fallback ke mock jika lib/hardware tidak ada
                self._real = False

    def read(self):
        if self._real:
            t = float(self._bme.temperature)   # °C
            h = float(self._bme.humidity)      # %
            p = float(self._bme.pressure)      # hPa
            return {"temp_c": t, "humidity": h, "pressure_hpa": p}

        # MOCK: variasi halus biar “hidup”
        dt = time.time() - self._t0
        temp = 28.0 + 1.0*math.sin(dt/3.0)
        hum  = 65.0 + 2.0*math.sin(dt/4.0 + 1.0)
        pres = 1008.0 + 1.5*math.sin(dt/5.0 + 2.0)
        return {
            "temp_c": round(temp, 2),
            "humidity": round(hum, 2),
            "pressure_hpa": round(pres, 2),
        }
