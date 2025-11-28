import unittest
import time
from src.plugins.hue.hue_plugin import HuePlugin

class TestHue(unittest.TestCase):
    def setUp(self):
        self.plugin = HuePlugin()
        time.sleep(2)

    def test_lights(self):
        data = self.plugin.get_data()
        self.assertIn("lights", data)
        self.assertGreater(len(data["lights"]), 0)

    def test_set_light(self):
        result = self.plugin.set_light("light_1", on=True, bri=200)
        self.assertEqual(result["success"], True)

if __name__ == '__main__':
    unittest.main(verbosity=2)
