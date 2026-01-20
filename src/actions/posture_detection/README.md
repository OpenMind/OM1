# Posture Detection and Reminder Skill

## Overview

The Posture Detection and Reminder Skill enables robots to monitor human posture in real-time and provide gentle reminders to maintain healthy posture. This is particularly useful for office workers, students, and anyone who spends long periods sitting or standing.

## Features

1. **Real-time Posture Monitoring**: Continuously monitors posture using computer vision
2. **Multiple Posture Types**: Detects slumped, hunched, leaning, asymmetric, and laying postures
3. **Severity Assessment**: Classifies posture issues as mild, moderate, or severe
4. **Duration Tracking**: Monitors how long someone has been in poor posture
5. **Gentle Reminders**: Provides encouraging, supportive reminders (configurable)
6. **Person Recognition**: Can identify specific individuals for personalized tracking
7. **Statistics Tracking**: Maintains posture statistics over time
8. **Recommendations**: Provides actionable advice to improve posture

## Architecture

### 1. Action Interface (`interface.py`)

Defines the `PostureDetection` action interface:
- `PostureType`: Enumeration of posture types (GOOD, SLUMPED, HUNCHED, LEANING, ASYMMETRIC, LAYING)
- `PostureSeverity`: Enumeration of severity levels (MILD, MODERATE, SEVERE)
- `PostureDetectionInput`: Input interface with posture type, severity, duration, person name, recommendations
- `PostureDetection`: Action interface definition

### 2. Connector (`connector/reminder.py`)

Implements reminder functionality:
- Records posture detections in HealthDetectionProvider
- Generates gentle, encouraging reminder messages
- Uses ElevenLabs TTS for voice reminders
- Implements reminder interval to avoid over-reminding
- Tracks reminder history per person

### 3. Health Detection Provider

The `HealthDetectionProvider` manages:
- Posture detection history
- Statistics and patterns
- Reminder intervals
- Per-person tracking

## Usage

### Configuration Example

```json5
{
  agent_inputs: [
    {
      type: "PostureDetectionInput",
      config: {
        posture_detection_base_url: "http://localhost:8080",
        poll_interval: 1.0,
      },
    },
  ],
  agent_actions: [
    {
      name: "posture_detection",
      llm_label: "posture_detection",
      connector: "reminder",
      config: {
        elevenlabs_api_key: "your_key",
        voice_id: "JBFqnCBsd6RMkjVDRZzb",
        reminder_interval_minutes: 30.0,
        enable_gentle_reminders: true,
      },
    },
  ],
}
```

### LLM Usage Examples

```python
# Slumped posture detected
PostureDetection: {
  'posture_type': 'slumped',
  'severity': 'moderate',
  'duration_minutes': 30,
  'person_name': 'Bob',
  'recommendation': 'Try standing up and stretching your back.'
}

# Good posture
PostureDetection: {
  'posture_type': 'good',
  'severity': 'mild'
}

# Hunched shoulders
PostureDetection: {
  'posture_type': 'hunched',
  'severity': 'severe',
  'duration_minutes': 45,
  'recommendation': 'Roll your shoulders back and lift your chin.'
}
```

## Posture Types

- **GOOD**: Healthy, aligned posture
- **SLUMPED**: Slouching forward, poor back support
- **HUNCHED**: Rounded shoulders, forward head position
- **LEANING**: Leaning to one side, uneven weight distribution
- **ASYMMETRIC**: Uneven posture, misaligned body
- **LAYING**: Person is laying down (may indicate fatigue)

## Reminder System

The reminder system is designed to be supportive and non-intrusive:

- **Gentle Mode** (default): Uses encouraging, friendly language
- **Direct Mode**: More straightforward reminders
- **Interval Control**: Configurable minimum time between reminders (default: 30 minutes)
- **Severity-based**: More urgent reminders for severe posture issues

## Integration with Posture Detection Service

The posture detection works with a dedicated posture detection service that provides structured posture data:

```json5
{
  agent_inputs: [
    {
      type: "PostureDetectionInput",
      config: {
        posture_detection_base_url: "http://localhost:8080",
        poll_interval: 1.0,
      },
    },
  ],
}
```

The `PostureDetectionInput` polls the posture detection service at `/posture/status` endpoint and provides structured posture information to the LLM, which then triggers the `PostureDetection` action when poor posture is detected.

### Posture Detection Service API

The posture detection service should provide a REST API endpoint:

- **GET /posture/status**: Returns current posture detection data:
  ```json
  {
    "posture_type": "slumped",
    "severity": "moderate",
    "duration_seconds": 1800.0,
    "person_name": "Bob",
    "recommendation": "Try standing up and stretching your back.",
    "confidence": 0.85
  }
  ```

The service can use computer vision, pose estimation, or other methods to detect posture.

## Health Benefits

Regular posture monitoring and reminders can help:
- Reduce back and neck pain
- Prevent long-term spinal issues
- Improve breathing and circulation
- Enhance overall well-being
- Build healthy posture habits

## Extension Suggestions

1. **Exercise Suggestions**: Provide specific stretches or exercises based on detected issues
2. **Ergonomic Recommendations**: Suggest desk/chair adjustments
3. **Progress Tracking**: Visualize posture improvement over time
4. **Break Reminders**: Suggest taking breaks when poor posture persists
5. **Posture Exercises**: Guide users through posture correction exercises
6. **Integration with Smart Furniture**: Connect with adjustable desks/chairs

## Notes

- Requires camera input for visual detection
- Works best with clear view of upper body
- May require calibration for different body types
- Consider privacy when monitoring posture
- Reminders are designed to be supportive, not nagging

