# Follow Person Action - Quick Start Guide

## Overview

The `follow_person` action enables a robot to identify and follow a specified person while maintaining a safe distance. This is useful for home assistance, tour guiding, or companion robot scenarios.

## Key Features

1. **Person Recognition**: Can follow a specific person by name, or follow the nearest/last seen person
2. **Safe Distance Control**: Configurable following distance with automatic safety maintenance
3. **Intelligent Obstacle Avoidance**: Automatically adjusts when person is too close or too far
4. **Status Feedback**: Real-time status feedback to the system

## File Structure

```
src/actions/follow_person/
├── interface.py              # Interface definition
├── connector/
│   ├── ros2.py              # ROS2 connector implementation
│   └── zenoh.py             # Zenoh connector implementation
└── implementation/
    └── passthrough.py       # Passthrough implementation
```

## Quick Setup

### 1. Register in Configuration File

In `config/your_agent.json5`:

```json5
{
  agent_actions: [
    {
      name: "follow_person",
      llm_label: "follow_person",
      implementation: "passthrough",
      connector: "ros2",
      config: {
        person_detection_topic: "/person_detection",
        movement_command_topic: "/cmd_vel",
        update_rate_hz: 10.0,
        max_following_distance: 5.0,
        min_following_distance: 0.8
      }
    },
  ],
}
```

### 2. Usage Examples

Users can command the robot using natural language:

- "Follow Alice" → Follow person named Alice
- "Follow the nearest person" → Follow the nearest person
- "Follow me" → Follow the last seen person
- "Follow Bob at 2 meters distance" → Follow Bob at 2 meters distance

## Technical Implementation Notes

### Required Components

1. **Person Detection System**
   - Requires VLM or dedicated person detection model
   - Must be able to identify and track specific persons
   - Output person position information (distance, angle)

2. **ROS2 Topics**
   - Subscribe: `/person_detection` - Receive person detection results
   - Publish: `/cmd_vel` - Send movement commands

3. **Control Algorithm**
   - PID controller for distance and angle control
   - Speed limits and safety checks
   - Timeout and loss handling

## Extension Suggestions

1. **Multi-person Tracking**: Support tracking multiple people simultaneously
2. **Path Planning**: Use navigation stack for path planning
3. **Gesture Recognition**: Recognize stop, accelerate gestures
4. **Voice Feedback**: Provide voice feedback during following
5. **Enhanced Obstacle Avoidance**: Integrate SLAM for better obstacle avoidance

## Testing Recommendations

1. Test following behavior at different distances
2. Test handling when person suddenly disappears/appears
3. Test selection logic in multi-person scenarios
4. Test following behavior in narrow spaces

## Notes

- Ensure person detection system is stable and reliable
- Set reasonable safety distances to avoid collisions
- Handle person detection failure cases
- Consider privacy and ethical issues (following functionality)

---

**Version**: 1.0.0  
**For complete documentation, see**: `follow_person_complete_documentation.md`

