# Pull Request: Add Follow Person Action

## Problem Statement

Currently, OM1 robots lack the ability to follow and track a specific person autonomously. This limits their usefulness in scenarios where:
- Robots need to follow a user around a space
- Personal assistance robots need to maintain proximity to their user
- Security or monitoring robots need to track specific individuals
- Interactive robots need to stay within interaction range

**Without this feature:**
- Robots cannot autonomously follow users
- Manual control is required to maintain proximity
- No mechanism exists for person-specific tracking and following
- Limited interaction capabilities in dynamic environments

## Solution

This PR adds a Follow Person action that enables robots to:
- Follow a specific person using computer vision and tracking
- Support multiple backends (ROS2 and Zenoh)
- Maintain appropriate following distance
- Handle person loss and re-acquisition
- Provide status updates to the LLM

## Type of Change
- [x] New feature (non-breaking change which adds functionality)
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] Documentation update

## Changes Made

### 1. FollowPerson Action Interface
**File:** `src/actions/follow_person/interface.py` (new)

**Implementation:**
- Defines `FollowPersonInput` dataclass for structured input
- Provides `FollowPerson` interface for action definition
- Supports person identification and following commands

### 2. ROS2 Connector
**File:** `src/actions/follow_person/connector/ros2.py` (new)

**Implementation:**
- Integrates with ROS2 for robot control
- Handles person following via ROS2 topics
- Manages following distance and speed
- Provides status feedback

### 3. Zenoh Connector
**File:** `src/actions/follow_person/connector/zenoh.py` (new)

**Implementation:**
- Integrates with Zenoh for robot control
- Handles person following via Zenoh messaging
- Manages following distance and speed
- Provides status feedback

### 4. Passthrough Implementation
**File:** `src/actions/follow_person/implementation/passthrough.py` (new)

**Implementation:**
- Simple passthrough implementation for basic following

### 5. Configuration Examples
**Files:**
- `config/follow_person_example.json5` (new)
- `config/follow_person_zenoh_example.json5` (new)

Provides complete configuration examples for both ROS2 and Zenoh backends.

### 6. Documentation
**Files:**
- `follow_person_complete_documentation.md` (new)
- `follow_person_summary.md` (new)
- `FOLLOW_PERSON_UPDATE_SUMMARY.md` (new)

Comprehensive documentation for the Follow Person feature.

## Testing

**Code Quality:**
- ? All code follows PEP 8 style guidelines
- ? Comprehensive docstrings for all classes and methods
- ? Type hints throughout
- ? No syntax errors

**Note:** Unit tests for Follow Person will be added in a follow-up PR to ensure comprehensive coverage.

## Files Changed

**New Files:**
- `src/actions/follow_person/interface.py`
- `src/actions/follow_person/connector/ros2.py`
- `src/actions/follow_person/connector/zenoh.py`
- `src/actions/follow_person/implementation/passthrough.py`
- `config/follow_person_example.json5`
- `config/follow_person_zenoh_example.json5`
- `follow_person_complete_documentation.md`
- `follow_person_summary.md`
- `FOLLOW_PERSON_UPDATE_SUMMARY.md`

## Benefits

- **Autonomous Following:** Enables robots to follow users without manual control
- **Multiple Backends:** Supports both ROS2 and Zenoh for flexibility
- **Distance Management:** Maintains appropriate following distance
- **Person Tracking:** Tracks specific individuals
- **Status Updates:** Provides feedback to LLM about following status

## Use Cases

- Personal assistance robots
- Security and monitoring applications
- Interactive robots in dynamic environments
- User-guided navigation scenarios

## Configuration

**ROS2 Example:**
```json5
{
  agent_actions: [
    {
      name: "follow_person",
      llm_label: "follow_person",
      connector: "ros2",
      config: {
        following_distance: 1.0,
        max_speed: 0.5,
      },
    },
  ],
}
```

**Zenoh Example:**
```json5
{
  agent_actions: [
    {
      name: "follow_person",
      llm_label: "follow_person",
      connector: "zenoh",
      config: {
        following_distance: 1.0,
        max_speed: 0.5,
      },
    },
  ],
}
```

## Checklist

- [x] Code follows the project's style guidelines
- [x] Self-review of code has been performed
- [x] Comments added for complex code
- [x] Documentation updated (included)
- [x] No new warnings generated
- [x] Changes are backward compatible
- [x] Problem statement clearly defined
- [x] All content in English only

## Notes

- Requires person detection and tracking system
- Works with ROS2 or Zenoh backend
- May require calibration for different robot platforms
- Following distance and speed are configurable
