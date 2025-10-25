# WeatherAgent 示例

这是一个为 [OpenMind OM1](https://github.com/openmind/OM1) 提供的简单示例智能体。  
该智能体能根据输入的城市名查询实时天气。

## 💡 功能概述

- 使用 OpenWeatherMap API 查询城市天气  
- 以自然语言输出当前气温与天气描述  
- 可作为入门示例学习如何继承 `Agent` 类  

## 🚀 运行步骤

1. 安装依赖并激活虚拟环境：
   ```bash
   uv venv && source .venv/bin/activate
   uv pip install requests
   ```

2. 设置 OpenWeather API Key：
   ```bash
   export OPENWEATHER_API_KEY="你的_openweather_api_key"
   ```

3. 运行示例：
   ```bash
   uv run examples/weather_agent.py
   ```

4. 输入城市名称，例如：
   ```
   北京
   ```
   输出：
   ```
   🌤 当前 北京 天气：晴朗，温度 21°C。
   ```

## 🧩 贡献类型

- 类型：Example
- 作者：你的名字
- 日期：2025-10-25
