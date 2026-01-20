# Pull Request: Add Posture Detection and Reminder Skill

## Problem Statement

Currently, OM1 robots lack the ability to monitor human posture and provide health-related reminders. This limits their usefulness in scenarios where:
- Office workers sit for long periods and need posture reminders
- Students studying at desks need gentle posture corrections
- People with back or neck pain need monitoring and encouragement
- Rehabilitation and physical therapy require posture tracking

**Without this feature:**
- Robots cannot help users maintain healthy posture
- Users miss opportunities for proactive health monitoring
- The system lacks health awareness capabilities
- No mechanism exists to track and improve posture over time

## Solution

This PR adds a comprehensive Posture Detection and Reminder Skill that:
- Detects multiple posture types (slumped, hunched, leaning, asymmetric, laying, good)
- Classifies posture severity (mild, moderate, severe)
- Tracks posture duration and provides gentle reminders
- Supports personalized tracking per person
- Maintains posture statistics over time
- Provides actionable recommendations

## Type of Change
- [x] New feature (non-breaking change which adds functionality)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] Documentation update

## Changes Made

### 1. PostureDetection Action Interface
**Files:**
- `src/actions/posture_detection/interface.py` (new)
- `src/actions/posture_detection/README.md` (new)

**Implementation:**
- Defines `PostureType` enum with 6 posture types (GOOD, SLUMPED, HUNCHED, LEANING, ASYMMETRIC, LAYING)
- Defines `PostureSeverity` enum with 3 severity levels (MILD, MODERATE, SEVERE)
- Provides `PostureDetectionInput` dataclass for structured input
- Implements `PostureDetection` interface for action definition

### 2. PostureReminderConnector
**File:** `src/actions/posture_detection/connector/reminder.py` (new)

**Implementation:**
- Handles posture detection and provides gentle reminders
- Integrates with ElevenLabs TTS for voice reminders
- Implements reminder interval logic to prevent over-reminding
- Supports both gentle and direct reminder modes
- Tracks reminder history per person
- Handles edge cases (TTS disabled, Zenoh unavailable, too many pending messages)

### 3. HealthDetectionProvider
**File:** `src/providers/health_detection_provider.py` (new)

**Implementation:**
- Manages posture detection history
- Provides posture statistics and pattern analysis
- Implements reminder interval management
- Supports per-person tracking

### 4. Unit Tests
**Files:**
- `tests/actions/posture_detection/test_interface.py` (new)
- `tests/actions/posture_detection/test_connector.py` (new)

**Test Coverage:**
- ? Posture classification logic (good vs poor postures)
- ? Reminder interval timing enforcement
- ? Edge cases:
  - TTS disabled
  - Zenoh unavailable (fallback to direct TTS)
  - Too many pending TTS messages
  - Camera unavailable scenarios (handled gracefully)
- ? Person-specific reminder tracking
- ? Severity-based message tone
- ? Gentle vs direct reminder modes

### 5. Configuration Example
**File:** `config/posture_detection_example.json5` (new)

Provides complete configuration example with:
- System prompts for posture monitoring
- VLM input configuration
- Posture detection action configuration
- Reminder settings

## Testing

All functionality has been thoroughly tested:

**Unit Tests:**
- ? Posture type and severity enum validation
- ? Posture classification logic (good vs poor)
- ? Reminder interval timing (respects configured intervals)
- ? Edge case handling:
  - TTS disabled → records but doesn't send reminder
  - Zenoh unavailable → falls back to direct TTS
  - Too many pending messages → skips to avoid queue overflow
- ? Person-specific reminder tracking
- ? Message generation for all posture types
- ? Severity-based message tone adjustment
- ? Gentle vs direct reminder mode differences

**Code Quality:**
- ? All code follows PEP 8 style guidelines
- ? Comprehensive docstrings for all classes and methods
- ? Type hints throughout
- ? No syntax errors
- ? All tests pass

## Files Changed

**New Files:**
- `src/actions/posture_detection/interface.py`
- `src/actions/posture_detection/connector/reminder.py`
- `src/actions/posture_detection/README.md`
- `src/providers/health_detection_provider.py`
- `config/posture_detection_example.json5`
- `tests/actions/posture_detection/__init__.py`
- `tests/actions/posture_detection/test_interface.py`
- `tests/actions/posture_detection/test_connector.py`

**Removed Files:**
- `FOLLOW_PERSON_更新总结.md` (Chinese documentation, removed per requirements)

## Benefits

- **Health Awareness:** Enables robots to monitor and improve user posture
- **Proactive Reminders:** Gentle, encouraging reminders help build healthy habits
- **Personalized Tracking:** Per-person tracking allows customized monitoring
- **Statistics:** Tracks posture patterns over time for insights
- **Configurable:** Flexible reminder intervals and modes
- **Robust:** Handles edge cases gracefully (TTS disabled, service unavailable, etc.)

## Use Cases

- Office workers who sit for long periods
- Students studying at desks
- People with back or neck pain
- Rehabilitation and physical therapy
- General health and wellness monitoring

## Configuration

```json5
{
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

## Checklist

- [x] Code follows the project's style guidelines
- [x] Self-review of code has been performed
- [x] Comments added for complex code
- [x] Documentation updated (README.md included)
- [x] No new warnings generated
- [x] Tests pass locally
- [x] Changes are backward compatible
- [x] Problem statement clearly defined
- [x] All content in English only
- [x] Unit tests cover classification logic, reminder intervals, and edge cases

## Notes

- Requires camera input via VLM for visual detection
- Works best with clear view of upper body
- May require calibration for different body types
- Reminders are designed to be supportive, not nagging
- Configurable reminder intervals prevent over-reminding
