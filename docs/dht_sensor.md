# DHT 环境传感器输入插件文档

## 概述
这个插件集成 DHT22 传感器，提供实时温度和湿度数据输入。AI 代理可使用这些数据做环境感知决策，例如：
- 如果温度 > 30°C，机器人暂停户外活动。
- 结合湿度，避免潮湿环境下的操作。

## 要求
- **硬件**：DHT22 传感器 + Raspberry Pi 或兼容 GPIO 设备。
- **依赖**：`adafruit-circuitpython-dht`（已加到 requirements.txt）。
- **兼容**：ROS2 或 Zenoh 接口（通过 HAL）。

## 安装
1. 连接 DHT22 到 GPIO 引脚（默认 D4）。
2. 更新依赖：`uv sync`。
3. 配置：在 config/ 文件添加 inputs 部分（见示例）。

## 配置示例
```json5
{
  "inputs": {
    "dht_sensor": {
      "type": "dht",
      "enabled": true,
      "pin": "D4"
    }
  }
}
