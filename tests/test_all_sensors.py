# tests/test_all_sensors.py
import time
import unittest

from src.plugins.sensors.temperature_plugin import TemperaturePlugin
from src.plugins.sensors.humidity_plugin import HumidityPlugin
from src.plugins.sensors.light_plugin import LightPlugin
from src.plugins.sensors.air_quality_plugin import AirQualityPlugin


class TestEnvSensors(unittest.TestCase):
    def test_temperature(self):
        p = TemperaturePlugin(interval=0.2)
        time.sleep(0.4)  # let it update
        data = p.get_data()
        self.assertIn("temperature", data)
        self.assertIn("comfort", data)

    def test_humidity(self):
        p = HumidityPlugin(interval=0.2)
        time.sleep(0.4)
        data = p.get_data()
        self.assertIn("humidity", data)
        self.assertIn("comfort", data)

    def test_light(self):
        p = LightPlugin(interval=0.2)
        time.sleep(0.4)
        data = p.get_data()
        self.assertIn("lux", data)
        self.assertIn("description", data)

    def test_air_quality(self):
        p = AirQualityPlugin(interval=0.2)
        time.sleep(0.4)
        data = p.get_data()
        self.assertIn("co2_ppm", data)
        self.assertIn("air_quality", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
