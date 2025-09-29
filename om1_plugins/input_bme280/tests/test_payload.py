# Author: Wanbogang (@Wanbogang)
import os
os.environ["MOCK"]="1"

from om1_plugins.input_bme280.config import load
from om1_plugins.input_bme280.driver_bme280 import BME280Driver

def test_driver_mock_once():
    cfg = load()
    d = BME280Driver(cfg)
    r = d.read()
    assert {"temp_c","humidity","pressure_hpa"} <= set(r.keys())
    assert isinstance(r["temp_c"], (int, float))
