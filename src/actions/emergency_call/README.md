# Emergency Call Plugin for OM1

A comprehensive emergency response system with multi-modal triggers and tiered response escalation.

## Features

🚨 **Multi-Modal Triggers**
- Voice keyword detection ("help", "emergency", "fall", etc.)
- IMU-based fall detection (accelerometer + gyroscope)
- Physical button triggers (single, double, long press)

📱 **Tiered Response System**
- **Tier 1**: Send notifications to family members (all levels)
- **Tier 2**: Initiate phone calls (MEDIUM and above)
- **Tier 3**: Contact emergency services (HIGH and CRITICAL)

🔒 **Privacy Protection**
- End-to-end encryption for logs
- Automatic deletion after 72 hours (configurable)
- Encrypted emergency data storage

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    EMERGENCY TRIGGERS                  │
├──────────────┬───────────────┬──────────────────────────┤
│   Voice      │    IMU        │   Physical Button        │
│  Keywords    │ Fall Detect   │   (Single/Double/Long)   │
└──────┬───────┴───────┬───────┴──────────┬───────────────┘
       │               │                  │
       └───────────────┴──────────────────┘
                       │
                       ▼
            ┌────────────────────┐
            │ EmergencyManager  │
            └────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ EmergencyCallConnector │
         └──────┬────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────────┐
│ Notify │ │ Phone  │ │ Emergency  │
│ Family │ │  Call  │ │  Services  │
└────────┘ └────────┘ └────────────┘
```

## Installation

### Requirements

```bash
pip install aiohttp cryptography numpy
```

Or add to `requirements.txt`:
```
aiohttp>=3.8.0
cryptography>=3.4.0
numpy>=1.21.0
```

## Configuration

Create `config/emergency_call.json5`:

```json5
{
  // Emergency Call Plugin Configuration
  "emergency_call": {
    // Encryption key for logs (keep secret!)
    "encryption_key": "your-secret-key-here",
    
    // Auto-delete logs after N hours
    "auto_delete_hours": 72,
    
    // Twilio credentials (optional, for phone calls)
    "twilio_account_sid": "your-twilio-sid",
    "twilio_auth_token": "your-twilio-token",
    "twilio_from_number": "+1234567890",
    
    // Emergency service number
    "emergency_service_number": "911",
    
    // Family contacts (priority order)
    "family_contacts": [
      {
        "name": "Spouse",
        "phone": "+1234567891",
        "email": "spouse@example.com",
        "relation": "spouse",
        "priority": 1
      },
      {
        "name": "Emergency Contact",
        "phone": "+1234567892",
        "email": "emergency@example.com",
        "relation": "emergency",
        "priority": 2
      }
    ]
  },
  
  "system": {
    "rate_limit": 1,  // Limit emergency calls
    "tick_rate": 10,
    "input": ["emergency_triggers"],
    "output": ["emergency_call"]
  }
}
```

## Usage

### Manual Emergency Trigger

```python
from actions.emergency_call import (
    EmergencyCallInput,
    EmergencyCallConnector,
    EmergencyCallConfig,
)
from actions.emergency_call.interface import (
    EmergencyLevel,
    EmergencyTriggerType,
)

# Create config
config = EmergencyCallConfig(
    encryption_key="secret",
    emergency_service_number="911",
    family_contacts=[...]
)

# Create connector
connector = EmergencyCallConnector(config)

# Trigger emergency
emergency = EmergencyCallInput(
    trigger_type=EmergencyTriggerType.FALL_DETECTION,
    emergency_level=EmergencyLevel.HIGH,
    location="kitchen",
    user_message="Fall detected by IMU sensor",
    sensor_data={"impact_g": 3.5, "fall_duration": 0.8}
)

await connector.connect(emergency)
```

### Using Triggers

```python
from triggers.emergency import EmergencyTriggerManager

# Create trigger manager
manager = EmergencyTriggerManager()

# Register callback
async def on_emergency(result, emergency_input):
    print(f"Emergency detected: {result.trigger_type}")
    # Handle emergency...

manager.register_callback(on_emergency)

# Start monitoring
await manager.start()

# Later: stop monitoring
await manager.stop()
```

## Trigger Details

### Voice Keyword Trigger

Detects emergency keywords in speech:
- Triggers: "help", "emergency", "fall", "fell", "hurt", "pain", "aid"
- Emergency Level: MEDIUM
- Confidence: Based on ASR confidence

### Fall Detection Trigger

Uses IMU sensors (accelerometer + gyroscope):
- Free fall detection (low g-force)
- Impact detection (high g-force)
- Inactivity detection after motion
- Emergency Level: HIGH

**Algorithm:**
1. Detect free fall (accel < 0.5g)
2. Detect impact (accel > 2.5g)
3. Check for inactivity after fall

### Physical Button Trigger

Supports multiple press patterns:
- **Double press**: Emergency triggered
- **Long press** (>3s): Emergency triggered
- **Single press**: No action (prevents accidents)
- Emergency Level: CRITICAL

## Privacy & Security

### Encryption

Logs are encrypted using Fernet (symmetric encryption):
- Key derived from config using PBKDF2
- Salt: `om1_emergency_salt`
- Iterations: 100,000

### Auto-Deletion

Logs older than `auto_delete_hours` are automatically removed.
Location: `~/.om1/emergency_logs/`

### Data Minimization

- Only essential data logged
- Sensor data encrypted at rest
- Timestamps encrypted
- Automatic cleanup prevents data accumulation

## Testing

Run tests:

```bash
pytest tests/actions/emergency_call/ -v
```

## Emergency Levels

| Level | Value | Response |
|-------|-------|----------|
| LOW | 1 | Notifications only |
| MEDIUM | 2 | Notifications + Phone calls |
| HIGH | 3 | All above + Emergency services |
| CRITICAL | 4 | Immediate emergency dispatch |

## Troubleshooting

### Encryption Errors

Ensure encryption key is set and non-empty:
```python
config = EmergencyCallConfig(encryption_key="your-key-here")
```

### Twilio Not Working

Check credentials are configured:
- `twilio_account_sid`
- `twilio_auth_token`
- `twilio_from_number`

### Logs Not Auto-Deleting

Verify `auto_delete_hours` is set correctly. Logs are checked at startup.

## License

MIT License - See LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.
