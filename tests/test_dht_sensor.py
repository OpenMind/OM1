# tests/test_dht_sensor.py
# 简单单元测试

import unittest
from omi.plugins.inputs.dht_sensor import DHTSensor

class TestDHTSensor(unittest.TestCase):
    def test_read_data(self):
        config = {"pin": "D4"}  # 模拟配置
        sensor = DHTSensor(config)
        data = sensor.read_data()
        self.assertIn("temperature", data)  # 检查有温度键
        self.assertIn("humidity", data)     # 检查有湿度键

if __name__ == '__main__':
    unittest.main()
