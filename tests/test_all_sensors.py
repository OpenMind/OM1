import unittest, time
from src.plugins.sensors.bme280_plugin import BME280Plugin
from src.plugins.sensors.dht22_plugin import DHT22Plugin
from src.plugins.sensors.bh1750_plugin import BH1750Plugin
from src.plugins.sensors.mq135_plugin import MQ135Plugin

class TestAllSensors(unittest.TestCase):
    def setUp(self):
        time.sleep(1)

    def test_bme280(self): p = BME280Plugin(); d = p.get_data(); self.assertIn("comfort", d)
    def test_dht22(self): p = DHT22Plugin(); d = p.get_data(); self.assertIn("comfort", d)
    def test_bh1750(self): p = BH1750Plugin(); d = p.get_data(); self.assertIn("description", d)
    def test_mq135(self): p = MQ135Plugin(); d = p.get_data(); self.assertIn("air_quality", d)

if __name__ == '__main__':
    unittest.main(verbosity=2)
