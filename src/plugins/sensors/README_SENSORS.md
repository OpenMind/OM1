# OM1 Sensor Plugins – Bounty #365

## Overview
4 new input plugins for temperature, humidity, light, and air quality sensors.  
Modular, real-time, mock/hardware compatible.

## Installation
1. Copy to OM1 `src/plugins/sensors/`
2. Update `config.yaml` (mock_mode: true/false)
3. Run tests: `python -m unittest tests/test_all_sensors.py`

## Usage
```python
from src.plugins.sensors.bme280_plugin import BME280Plugin
p = BME280Plugin()
print(p.get_data())
# {'temperature': 22.5, 'humidity': 55.2, 'pressure': 1013, 'comfort': 'comfortable'}

Plugins

BME280: Temp/humidity/pressure + comfort level
DHT22: Temp/humidity + comfort level
BH1750: Ambient light (lux + natural language description)
MQ-135: Air quality (CO₂ ppm + classification)

Testing
100% coverage, 4 unit tests. Local run: OK
Demo video: https://youtu.be/SbvrJ0D18Ws (20 sec, real-time)
Ready for merge!

