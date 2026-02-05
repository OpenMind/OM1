## Summary
This PR adds Home Assistant integration to OM1, enabling robots to control and monitor smart home devices.

## 🤖 Home Assistant Integrated
- REST API connector (primary)
- MQTT connector (alternative)

## Features
### Action Plugin (`home_assistant`)
- **Lights**: on/off, toggle, brightness (0-255), RGB color
- **Switches**: on/off, toggle  
- **Thermostats**: set temperature, HVAC modes (heat/cool/auto/off)
- **Covers**: open/close/stop (blinds, garage doors)
- **Fans**: on/off, toggle

### Input Plugin (`HomeAssistantStateInput`)
- Real-time device state monitoring
- Change detection (only report when states change)
- Natural language formatting for LLM consumption
- Supports: lights, switches, climate, sensors, binary sensors, covers, fans

## Files Changed
- `src/actions/home_assistant/interface.py` - Action interface definitions
- `src/actions/home_assistant/connector/rest_api.py` - REST API connector
- `src/actions/home_assistant/connector/mqtt.py` - MQTT connector
- `src/inputs/plugins/home_assistant_state.py` - State monitoring input
- `tests/actions/test_home_assistant.py` - Action tests (30+ test cases)
- `tests/inputs/test_home_assistant_state.py` - Input tests
- `config/home_assistant.json5` - Demo configuration
- `docs/smart_devices/HOME_ASSISTANT_SETUP.md` - Setup documentation

## Configuration Example
```json5
{
  agent_actions: [{
    name: "home_assistant",
    llm_label: "smart_home",
    connector: "rest_api",
    config: {
      base_url: "http://homeassistant.local:8123",
      access_token: "YOUR_TOKEN"
    }
  }],
  agent_inputs: [{
    type: "HomeAssistantStateInput",
    config: {
      base_url: "http://homeassistant.local:8123",
      access_token: "YOUR_TOKEN",
      entity_ids: ["light.living_room", "climate.bedroom"]
    }
  }]
}
```

## 🎥 Demo Video
Coming soon - will record once PR is reviewed.

## 📑 Notes
- **Setup**: Get long-lived token from Home Assistant Profile → Long-Lived Access Tokens
- **Limitations**: MQTT connector requires aiomqtt package
- **Improvements**: Could add support for scenes, automations, and scripts

Closes #366
