# Home Assistant Action for OM1

This action module enables OM1 to control smart home devices via Home Assistant REST API.

## Features

- **Light Control**: Turn on/off, set brightness (0-255), set RGB color
- **Switch Control**: Turn on/off, toggle
- **Thermostat Control**: Set target temperature
- **State Query**: Get current device state

## Requirements

- Home Assistant instance with API enabled
- Long-lived access token (generate from HA Profile page)
- Network access to Home Assistant (default port 8123)

## Configuration

Add to your agent config (e.g., `config/your_agent.json5`):

```json5
{
  "actions": [
    {
      "name": "home_assistant",
      "llm_label": "HOME",
      "interface": "HomeAssistant",
      "connector": "HomeAssistantAPIConnector",
      "config": {
        "base_url": "http://192.168.1.100:8123",
        "access_token": "YOUR_LONG_LIVED_ACCESS_TOKEN",
        "verify_ssl": true,
        "timeout": 10
      }
    }
  ]
}
```

### Getting a Long-Lived Access Token

1. Open Home Assistant web interface
2. Click on your profile (bottom left)
3. Scroll to "Long-Lived Access Tokens"
4. Click "Create Token"
5. Copy the token (it won't be shown again!)

## Usage Examples

### Turn on a light

```python
from actions.home_assistant import HomeAssistantInput

input_data = HomeAssistantInput(
    action="turn_on",
    entity_id="light.living_room"
)
```

### Set light brightness

```python
input_data = HomeAssistantInput(
    action="set_brightness",
    entity_id="light.bedroom",
    brightness=128  # 0-255
)
```

### Set light color

```python
input_data = HomeAssistantInput(
    action="set_color",
    entity_id="light.desk",
    rgb_color=(255, 128, 0)  # Orange
)
```

### Set thermostat temperature

```python
input_data = HomeAssistantInput(
    action="set_temperature",
    entity_id="climate.thermostat",
    temperature=22.0
)
```

### Get device state

```python
input_data = HomeAssistantInput(
    action="get_state",
    entity_id="light.living_room"
)
```

### Toggle a switch

```python
input_data = HomeAssistantInput(
    action="toggle",
    entity_id="switch.desk_lamp"
)
```

## Supported Actions

| Action | Domain | Parameters | Description |
|--------|--------|------------|-------------|
| `turn_on` | any | entity_id | Turn on device |
| `turn_off` | any | entity_id | Turn off device |
| `toggle` | any | entity_id | Toggle device state |
| `set_brightness` | light | entity_id, brightness (0-255) | Set light brightness |
| `set_color` | light | entity_id, rgb_color (r,g,b) | Set light RGB color |
| `set_temperature` | climate | entity_id, temperature | Set thermostat target |
| `get_state` | any | entity_id | Query current state |

## Entity ID Format

Home Assistant uses `domain.name` format for entity IDs:

- `light.living_room` - A light entity
- `switch.fan` - A switch entity
- `climate.thermostat` - A climate/thermostat entity
- `sensor.temperature` - A sensor entity

Find your entity IDs in Home Assistant:
1. Go to Settings -> Devices & Services -> Entities
2. Or use Developer Tools -> States

## Troubleshooting

### Connection Refused
- Check if Home Assistant is running
- Verify the `base_url` is correct
- Check if API is enabled in HA config

### 401 Unauthorized
- Verify your access token is correct
- Make sure token hasn't expired
- Generate a new long-lived token

### 404 Not Found
- Check if the entity_id exists in Home Assistant
- Verify the entity_id format (domain.name)

### SSL Certificate Errors
- Set `verify_ssl: false` for self-signed certificates
- Or install proper SSL certificates

## License

MIT License - See [LICENSE](../../LICENSE)
