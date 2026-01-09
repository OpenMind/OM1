# src/omi/plugins/inputs/dht_sensor.py
# DHT 环境传感器输入插件：读取温度和湿度数据
# 支持 DHT22 传感器，通过 GPIO 连接（e.g., Raspberry Pi）

import adafruit_dht  # 导入 DHT 库
import board  # 用于 GPIO 引脚
from omi.core.plugin import InputPlugin  # OM1 输入插件基类（根据项目调整，如果路径不同）

class DHTSensor(InputPlugin):
    def __init__(self, config):
        super().__init__(config)
        # 从配置读取引脚号，默认用 D4 (GPIO4)
        pin = config.get('pin', board.D4)
        self.sensor = adafruit_dht.DHT22(pin)
        self.use_pulseio = config.get('use_pulseio', False)  # 可选参数

    def read_data(self):
        # 读取温度和湿度
        try:
            temperature = self.sensor.temperature
            humidity = self.sensor.humidity
            return {
                "temperature": temperature,  # 摄氏度
                "humidity": humidity  # 百分比
            }
        except RuntimeError as error:
            # DHT 传感器有时读失败，重试
            return {"error": str(error)}
        except Exception as error:
            return {"error": str(error)}

    def cleanup(self):
        # 清理资源
        self.sensor.exit()

# 注册插件（在 OM1 主代码中添加导入，或用插件注册机制）
# 示例：在 omi/plugins/__init__.py 加：from .inputs.dht_sensor import DHTSensor
# register_plugin('dht_sensor', DHTSensor)
