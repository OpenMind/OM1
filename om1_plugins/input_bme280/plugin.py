# Author: Wanbogang (@Wanbogang) — Contribution for Issue #365 (OM1 Bounty Program)
# License: MIT

import json, time, sys, signal
from .config import load
from .driver_bme280 import BME280Driver

_STOP = False
def _handle_stop(signum, frame):
    global _STOP
    _STOP = True

def main():
    cfg = load()
    drv = BME280Driver(cfg)

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    # Bila output dipotong (mis. `head`), set SIGPIPE default → exit rapi (hindari spam Broken pipe)
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        pass

    interval = max(0.05, cfg["poll_ms"] / 1000.0)
    while not _STOP:
        try:
            data = drv.read()
            ev = {
                "type": "sensor.bme280",
                "sensor_id": cfg["sensor_id"],
                "ts": int(time.time() * 1000),
                "data": data,
            }
            print(json.dumps(ev, ensure_ascii=False))
            sys.stdout.flush()
        except Exception as e:
            print(f"[warn] read failed: {e}", file=sys.stderr)
        time.sleep(interval)

if __name__ == "__main__":
    main()
