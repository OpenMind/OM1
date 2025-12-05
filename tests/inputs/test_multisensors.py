import pytest
from src.inputs.env_sensors.temperature import TemperatureSensor
from src.inputs.env_sensors.humidity import HumiditySensor
from src.inputs.env_sensors.light import LightSensor

def test_temperature_read():
    sensor = TemperatureSensor()
    sensor.start()
    data = sensor.read()
    assert data["source"] == "temperature"
    assert 20.0 <= data["payload"]["temperature_c"] <= 30.0
    sensor.stop()

def test_humidity_read():
    sensor = HumiditySensor()
    sensor.start()
    data = sensor.read()
    assert data["source"] == "humidity"
    assert "humidity_percent" in data["payload"]
    sensor.stop()

def test_light_read():
    sensor = LightSensor()
    sensor.start()
    data = sensor.read()
    assert data["source"] == "light"
    assert data["payload"]["lux"] > 0
    sensor.stop()
