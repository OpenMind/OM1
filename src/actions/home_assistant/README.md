# Home Assistant Integration for OM1

This module integrates OM1 with Home Assistant to control IoT devices including lights, switches, and thermostats.

## Features

- ✅ Control lights (on/off, brightness, color)
- ✅ Control switches (on/off)
- ✅ Control thermostats (set temperature)
- ✅ REST API communication with Home Assistant
- ✅ Async support for non-blocking operations

## Setup

### 1. Configure Home Assistant

Enable the REST API in your Home Assistant instance. The API is enabled by default, but you need a long-lived access token.

### 2. Get Access Token

1. Open Home Assistant UI
2. Click on your user profile (bottom left)
3. Scroll to "Long-Lived Access Tokens"
4. Click "Create Token"
5. Give it a name (e.g., "OM1 Robot")
6. **Copy the token immediately** (it won't be shown again)

### 3. Configure OM1

Create a config file (e.g., `home_assistant.json5`):

```json5
{
  "home_assistant": {
    "base_url": "http://homeassistant.local:8123",
    "token": "YOUR_TOKEN_HERE"
  },
  
  "system": {
    "input": ["home_assistant"],
    "output": ["home_assistant"]
  }
}
```

### 4. Run OM1 with Home Assistant

```bash
python -m om1 home_assistant.json5
```

## Usage Examples

### Turn On a Light

```python
from actions.home_assistant.interface import (
    HomeAssistantInput,
    HomeAssistantAction,
    HomeAssistantDeviceType,
)

input_data = HomeAssistantInput(
    device_type=HomeAssistantDeviceType.LIGHT,
    device_id="light.living_room",
    action=HomeAssistantAction.TURN_ON,
    brightness=255,  # 0-255
    color="blue"
)
```

### Turn Off a Switch

```python
input_data = HomeAssistantInput(
    device_type=HomeAssistantDeviceType.SWITCH,
    device_id="switch.kitchen",
    action=HomeAssistantAction.TURN_OFF,
)
```

### Set Thermostat Temperature

```python
input_data = HomeAssistantInput(
    device_type=HomeAssistantDeviceType.THERMOSTAT,
    device_id="climate.bedroom",
    action=HomeAssistantAction.SET_TEMPERATURE,
    temperature=22.5,  # Celsius
)
```

## Testing

Run the tests:

```bash
pytest tests/actions/home_assistant/ -v
```

## Troubleshooting

### Connection Refused

- Verify Home Assistant is running
- Check the `base_url` is correct
- Ensure Home Assistant is accessible from the OM1 machine

### 401 Unauthorized

- Check your access token is correct
- Verify the token hasn't expired
- Create a new token if needed

### Device Not Found

- Verify the entity ID exists in Home Assistant
- Check Developer Tools > States in Home Assistant
- Use the exact entity ID (e.g., `light.living_room` not just `living_room`)

## License

MIT License - See LICENSE file for details.
