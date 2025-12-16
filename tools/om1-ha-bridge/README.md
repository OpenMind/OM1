# OM1 ↔ Home Assistant MQTT Bridge

A lightweight bridge that connects OM1 to Home Assistant using MQTT Discovery for light and climate (thermostat) control.

## Features

- **MQTT Discovery**: Automatic device configuration in Home Assistant
- **Light Control**: On/off, brightness (0-255), and HS color control
- **Climate Control**: Thermostat with mode and temperature control
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
# MQTT Broker
MQTT_HOST=localhost          # Your MQTT broker address
MQTT_PORT=1883              # MQTT broker port
MQTT_USER=                  # (Optional) MQTT username
MQTT_PASS=                  # (Optional) MQTT password
HA_DISCOVERY_PREFIX=homeassistant

# Light Device
DEVICE_ID=om1_light_1
DEVICE_NAME="OM1 Light 1"

# Thermostat Device
THERMO_DEVICE_ID=om1_thermo_1
THERMO_DEVICE_NAME="OM1 Thermostat 1"
MIN_TEMP=16                 # Minimum temperature (°C)
MAX_TEMP=30                 # Maximum temperature (°C)
TEMP_STEP=0.5               # Temperature adjustment step

# API Server
HOST=0.0.0.0
PORT=8081
```

## API Endpoints

### Light Control

#### Turn Light On
```bash
curl -X POST http://localhost:8081/light/on
```

#### Turn Light Off
```bash
curl -X POST http://localhost:8081/light/off
```

#### Set Brightness (0-255)
```bash
curl -X POST http://localhost:8081/light/brightness \
  -H "Content-Type: application/json" \
  -d '{"value": 128}'
```

#### Set Color (HS: Hue 0-360, Saturation 0-100)
```bash
curl -X POST http://localhost:8081/light/color \
  -H "Content-Type: application/json" \
  -d '{"hs": [120, 100]}'
```

### Climate (Thermostat) Control

#### Set Mode (off/heat/cool/auto)
```bash
curl -X POST http://localhost:8081/climate/mode \
  -H "Content-Type: application/json" \
  -d '{"value": "heat"}'
```

#### Set Target Temperature
```bash
curl -X POST http://localhost:8081/climate/target \
  -H "Content-Type: application/json" \
  -d '{"value": 24.5}'
```

#### Update Current Temperature
```bash
curl -X POST http://localhost:8081/climate/current \
  -H "Content-Type: application/json" \
  -d '{"value": 26.0}'
```

## How It Works

1. Bridge starts and connects to MQTT broker
2. Publishes Home Assistant MQTT Discovery messages for:
   - Light entity (with brightness and color support)
   - Climate entity (thermostat with mode and temperature)
3. Devices appear automatically in Home Assistant
4. HTTP API receives commands from OM1
5. Commands are forwarded to Home Assistant via MQTT

## MQTT Topics

### Light
- Discovery: `homeassistant/light/{DEVICE_ID}/config`
- State: `homeassistant/light/{DEVICE_ID}/state`
- Command: `homeassistant/light/{DEVICE_ID}/set`
- Brightness: `homeassistant/light/{DEVICE_ID}/brightness`
- Color: `homeassistant/light/{DEVICE_ID}/hs`

### Climate
- Discovery: `homeassistant/climate/{THERMO_DEVICE_ID}/config`
- Mode: `homeassistant/climate/{THERMO_DEVICE_ID}/mode`
- Target Temp: `homeassistant/climate/{THERMO_DEVICE_ID}/target_temp`
- Current Temp: `homeassistant/climate/{THERMO_DEVICE_ID}/current_temp`

## Requirements

- Python 3.7+
- MQTT broker (e.g., Mosquitto)
- Home Assistant with MQTT integration enabled

## Dependencies

This PR complements [PR #442](https://github.com/OpenMind/OM1/pull/442) (light control). Please merge #442 first.

## Notes

- Temperature limits are controlled by `MIN_TEMP`, `MAX_TEMP`, `TEMP_STEP` (defaults: 16-30°C, step 0.5)
- Supported modes: `off`, `heat`, `cool`, `auto`
- State topics are subscribed and logged; publishing state back to OM1 can be added in a follow-up
- Scope kept minimal per bounty requirements
