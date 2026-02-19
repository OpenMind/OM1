## Summary

Implements the Emergency Call Plugin requested in #2145 with multi-modal triggers and tiered response escalation.

## Features Implemented

🚨 **Multi-Modal Triggers**
- **Voice Keyword**: Detects "help", "emergency", "fall", "fell", "hurt", "pain", "aid"
- **IMU Fall Detection**: Accelerometer + gyroscope based fall detection with free fall + impact detection
- **Physical Button**: Single, double (emergency), and long press (>3s) patterns

📱 **Tiered Response System**
- **Tier 1**: Send notifications to family (all levels)
- **Tier 2**: Initiate phone calls via Twilio (MEDIUM+)
- **Tier 3**: Contact emergency services (HIGH/CRITICAL)

🔒 **Privacy Protection**
- Fernet end-to-end encryption for logs
- PBKDF2 key derivation (100k iterations)
- Auto-deletion after 72 hours (configurable)
- Encrypted storage at `~/.om1/emergency_logs/`

## Files Added

```
src/actions/emergency_call/
├── __init__.py
├── interface.py              # EmergencyLevel, EmergencyTriggerType enums
├── connector/
│   ├── __init__.py
│   └── emergency_call_connector.py  # Main connector with tiered response
└── README.md                 # Full documentation

src/triggers/emergency/
└── __init__.py               # Voice, Fall, Button trigger implementations

config/emergency_call.json5   # Example configuration
Dockerfile.emergency          # Simulation environment

tests/actions/emergency_call/
└── test_emergency_call.py    # Unit tests
```

## Usage Example

```python
from actions.emergency_call import EmergencyCallInput, EmergencyLevel
from actions.emergency_call.interface import EmergencyTriggerType

# Trigger HIGH level emergency (fall detection)
emergency = EmergencyCallInput(
    trigger_type=EmergencyTriggerType.FALL_DETECTION,
    emergency_level=EmergencyLevel.HIGH,
    location="kitchen",
    user_message="Fall detected by IMU sensor",
    sensor_data={"impact_g": 3.5, "fall_duration": 0.8}
)

await connector.connect(emergency)
# → Notifications sent
# → Phone calls initiated
# → Emergency services contacted
```

## Configuration

```json5
{
  emergency_call: {
    encryption_key: "your-secret-key",
    auto_delete_hours: 72,
    twilio_account_sid: "...",
    twilio_auth_token: "...",
    twilio_from_number: "+1234567890",
    emergency_service_number: "911",
    family_contacts: [
      {name: "Spouse", phone: "+123", email: "...", priority: 1}
    ]
  }
}
```

## Testing

```bash
pytest tests/actions/emergency_call/ -v
```

## Checklist

- [x] Modular implementation
- [x] Clear documentation
- [x] Unit tests
- [x] Docker simulation environment
- [ ] Demo video (to be added)

## Privacy Note

All emergency logs are encrypted and auto-deleted after 72 hours to protect user privacy.

closes #2145