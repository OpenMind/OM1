# OM1 ↔ Home Assistant MQTT Bridge

A lightweight bridge that connects OM1 to Home Assistant using MQTT Discovery for light control.

## Features

- **MQTT Discovery**: Automatic device configuration in Home Assistant
- **Light Control**: On/off, brightness (0-255), and HS color control
- **Simple HTTP API**: Easy integration with OM1
- **Environment-based Config**: Secure credential management

## Quick Start
```bash
cd tools/om1-ha-bridge
cp env.sample .env
# Edit .env with your MQTT broker details
./run.sh
```

## Configuration

Copy `env.sample` to `.env` and configure:
```bash
MQTT_HOST=localhost          # Your MQTT broker address
MQTT_PORT=1883              # MQTT broker port
MQTT_USER=                  # (Optional) MQTT username
MQTT_PASS=                  # (Optional) MQTT password
HA_DISCOVERY_PREFIX=homeassistant
DEVICE_ID=om1_light_1
DEVICE_NAME="OM1 Light 1"
HOST=0.0.0.0
PORT=8081
```

## API Endpoints

### Turn Light On
```bash
curl -X POST http://localhost:8081/light/on
```

### Turn Light Off
```bash
curl -X POST http://localhost:8081/light/off
```

### Set Brightness (0-255)
```bash
curl -X POST http://localhost:8081/light/brightness \
  -H "Content-Type: application/json" \
  -d '{"value": 128}'
```

### Set Color (HS: Hue 0-360, Saturation 0-100)
```bash
curl -X POST http://localhost:8081/light/color \
  -H "Content-Type: application/json" \
  -d '{"hs": [120, 100]}'
```

## How It Works

1. Bridge starts and connects to MQTT broker
2. Publishes Home Assistant MQTT Discovery message
3. Light appears automatically in Home Assistant
4. HTTP API receives commands from OM1
5. Commands are forwarded to Home Assistant via MQTT

## Requirements

- Python 3.7+
- MQTT broker (e.g., Mosquitto)
- Home Assistant with MQTT integration enabled

## Notes

- Scope kept minimal per bounty: single light entity
- State topics are subscribed and logged for future follow-up
- Can be extended for multiple devices or additional entity types
