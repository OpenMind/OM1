# Home Assistant Integration for OM1

This module provides integration between OM1 and [Home Assistant](https://www.home-assistant.io/), enabling OM1-powered robots to control and monitor smart home devices.

## Features

- **Device Control (Action Plugin)**: Control lights, switches, thermostats, covers, and fans
- **Device Monitoring (Input Plugin)**: Monitor device states and react to changes
- **Multiple Protocols**: Support for REST API and MQTT communication
- **LLM-Friendly Interface**: Natural language descriptions for device states

## Supported Devices

| Device Type | Actions | Features |
|-------------|---------|----------|
| **Lights** | on/off, toggle, brightness, color | RGB color support, brightness 0-255 |
| **Switches** | on/off, toggle | Basic on/off control |
| **Thermostats** | set_temperature, set_hvac_mode | Temperature, HVAC modes (heat/cool/auto) |
| **Covers** | open/close, stop | Blinds, garage doors |
| **Fans** | on/off, toggle | Basic fan control |

## Installation

### Prerequisites

1. **Home Assistant** running and accessible
2. **Long-Lived Access Token** from Home Assistant
   - Go to Profile → Long-Lived Access Tokens → Create Token

### Dependencies

Add to your `pyproject.toml`:

```toml
dependencies = [
    "aiohttp>=3.8.0",
    "aiomqtt>=2.0.0",  # Optional: for MQTT connector
]
```

## Configuration

### Action Plugin (Control Devices)

Add to your OM1 config file (e.g., `config/home_assistant.json5`):

```json5
{
  agent_actions: [
    {
      name: "home_assistant",
      llm_label: "smart_home",
      connector: "rest_api",
      config: {
        base_url: "http://homeassistant.local:8123",
        access_token: "YOUR_LONG_LIVED_TOKEN",
        verify_ssl: false,
        timeout: 10
      }
    }
  ]
}
```

### Input Plugin (Monitor Devices)

Add to your `agent_inputs`:

```json5
{
  agent_inputs: [
    {
      type: "HomeAssistantStateInput",
      config: {
        base_url: "http://homeassistant.local:8123",
        access_token: "YOUR_LONG_LIVED_TOKEN",
        entity_ids: [
          "light.living_room",
          "switch.coffee_maker",
          "climate.bedroom",
          "sensor.temperature"
        ],
        poll_interval: 5.0,
        report_all_states: false,
        input_name: "Smart Home Devices"
      }
    }
  ]
}
```

## Full Example Configuration

```json5
{
  version: "v1.0.1",
  hertz: 1,
  name: "smart_home_robot",
  api_key: "openmind_free",
  
  system_prompt_base: "You are a helpful home assistant robot. You can control smart home devices and respond to their state changes. When someone asks you to control a device, use the smart_home action.",
  
  system_governance: "Be helpful and safe. Only control devices when explicitly asked.",
  
  system_prompt_examples: "Examples:\n1. 'Turn on the living room light'\n   smart_home: {device_type: 'light', entity_id: 'light.living_room', action: 'turn_on'}\n\n2. 'Set bedroom to 22 degrees'\n   smart_home: {device_type: 'climate', entity_id: 'climate.bedroom', action: 'set_temperature', temperature: 22}",
  
  agent_inputs: [
    {
      type: "HomeAssistantStateInput",
      config: {
        base_url: "http://homeassistant.local:8123",
        access_token: "${HASS_TOKEN}",
        entity_ids: [
          "light.living_room",
          "light.bedroom",
          "switch.coffee_maker",
          "climate.bedroom",
          "sensor.living_room_temperature"
        ],
        poll_interval: 5.0,
        report_all_states: false,
        input_name: "Smart Home"
      }
    }
  ],
  
  agent_actions: [
    {
      name: "home_assistant",
      llm_label: "smart_home",
      connector: "rest_api",
      config: {
        base_url: "http://homeassistant.local:8123",
        access_token: "${HASS_TOKEN}",
        verify_ssl: false,
        timeout: 10
      }
    },
    {
      name: "speak",
      llm_label: "speak",
      connector: "elevenlabs_tts",
      config: {
        elevenlabs_api_key: "${ELEVENLABS_KEY}"
      }
    }
  ],
  
  cortex_llm: {
    type: "OpenAILLM",
    config: {
      agent_name: "HomeBot",
      history_length: 10
    }
  }
}
```

## Action Interface

The `smart_home` action accepts the following parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `device_type` | string | Yes | Device type: `light`, `switch`, `climate`, `cover`, `fan` |
| `entity_id` | string | Yes | Home Assistant entity ID (e.g., `light.living_room`) |
| `action` | string | Yes | Action to perform (see table below) |
| `brightness` | int | No | Brightness level 0-255 (for lights) |
| `color_rgb` | string | No | RGB color as "R,G,B" (e.g., "255,0,0" for red) |
| `temperature` | float | No | Target temperature (for thermostats) |
| `hvac_mode` | string | No | HVAC mode: `off`, `heat`, `cool`, `auto`, `dry` |

### Actions by Device Type

| Device Type | Available Actions |
|-------------|-------------------|
| Light | `turn_on`, `turn_off`, `toggle`, `brightness`, `color` |
| Switch | `turn_on`, `turn_off`, `toggle` |
| Climate | `set_temperature`, `set_hvac_mode`, `turn_off` |
| Cover | `open`, `close`, `stop`, `toggle` |
| Fan | `turn_on`, `turn_off`, `toggle` |

## MQTT Connector (Alternative)

For MQTT-based control, use the `mqtt` connector:

```json5
{
  name: "home_assistant",
  llm_label: "smart_home",
  connector: "mqtt",
  config: {
    mqtt_host: "localhost",
    mqtt_port: 1883,
    mqtt_username: "user",
    mqtt_password: "password",
    topic_prefix: "homeassistant"
  }
}
```

## Getting Your Home Assistant Token

1. Open Home Assistant web interface
2. Click on your profile (bottom left)
3. Scroll to "Long-Lived Access Tokens"
4. Click "Create Token"
5. Name it (e.g., "OM1 Robot")
6. Copy the token immediately (it won't be shown again)

## Finding Entity IDs

1. Go to Home Assistant → Developer Tools → States
2. Find your device in the list
3. The entity ID is shown (e.g., `light.living_room`)

## Troubleshooting

### Connection Refused
- Verify Home Assistant is running and accessible
- Check the base_url (include port 8123)
- Ensure no firewall blocking connections

### Authentication Failed
- Verify your access token is correct
- Check token hasn't expired
- Regenerate token if needed

### Entity Not Found
- Verify the entity_id exists in Home Assistant
- Check Developer Tools → States for correct ID
- Entity IDs are case-sensitive

### SSL Errors
- For local network, set `verify_ssl: false`
- For remote access, use proper SSL certificates

## API Reference

### HomeAssistantInput

```python
@dataclass
class HomeAssistantInput:
    device_type: str      # light, switch, climate, cover, fan
    entity_id: str        # Home Assistant entity ID
    action: str           # Action to perform
    brightness: Optional[int] = None      # 0-255 for lights
    color_rgb: Optional[str] = None       # "R,G,B" format
    temperature: Optional[float] = None   # For thermostats
    hvac_mode: Optional[str] = None       # off, heat, cool, auto
```

### HomeAssistantOutput

```python
@dataclass
class HomeAssistantOutput:
    success: bool         # Whether action succeeded
    message: str          # Response message
    entity_id: str        # Controlled entity
    new_state: Optional[str] = None  # New device state
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
