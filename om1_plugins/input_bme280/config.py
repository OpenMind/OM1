# Author: Wanbogang (@Wanbogang) — Contribution for Issue #365 (OM1 Bounty Program)
# License: MIT

import os

def _parse_addr(v: str) -> int:
    v = str(v).strip()
    return int(v, 16) if v.lower().startswith("0x") else int(v)

def load():
    """Parse konfigurasi dari ENV dengan default aman."""
    return {
        "sensor_type": os.getenv("SENSOR_TYPE", "bme280"),
        "sensor_id":  os.getenv("SENSOR_ID",  "bme280-1"),
        "i2c_bus":    int(os.getenv("I2C_BUS", "1")),
        "i2c_addr":   _parse_addr(os.getenv("I2C_ADDR", "0x76")),
        "poll_ms":    int(os.getenv("POLL_MS", "1000")),
        "mock":       os.getenv("MOCK", "0").lower() in ("1","true","yes"),
    }
