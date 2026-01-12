# Follow Person Action - Complete Documentation

## Overview

`follow_person` is a complete robot action that allows the robot to identify and follow a specified person while maintaining a safe distance. The action supports multiple following modes, including following by name, following the nearest person, and following the last seen person.

## Features

### Core Features
- ✅ **Multiple Following Modes**: By name, nearest person, last seen person
- ✅ **Safe Distance Control**: Configurable following distance with automatic safety range maintenance
- ✅ **Intelligent Obstacle Avoidance**: Automatic adjustment when person is too close or too far
- ✅ **Timeout Protection**: Automatic stop when person is lost for more than the configured time
- ✅ **Real-time Status Feedback**: Real-time status via `FollowPersonStatus` input
- ✅ **Multi-platform Support**: Supports both ROS2 and Zenoh communication methods

### Technical Features
- Smooth following based on PID control algorithm
- Asynchronous control loop that doesn't block the main event loop
- Thread-safe state management
- Comprehensive error handling and logging

## File Structure

```
src/actions/follow_person/
├── interface.py              # Interface definition (input/output structure)
├── connector/
│   ├── ros2.py              # ROS2 connector implementation
│   └── zenoh.py             # Zenoh connector implementation
└── implementation/
    └── passthrough.py       # Passthrough implementation (no additional business logic)
```

## Interface Definition

### FollowPersonInput

```python
@dataclass
class FollowPersonInput:
    action: str              # Person identifier or mode
    distance: float = 1.5    # Following distance (meters)
    speed: float = 0.5       # Following speed (0.0-1.0)
    stop_on_arrival: bool = True  # Whether to stop when reaching target distance
    timeout_sec: float = 30.0      # Timeout duration (seconds)
```

### Supported Action Values

- **Follow by name**: `"alice"`, `"bob"`, `"wendy"`, etc. (registered person names)
- **Follow nearest person**: `"nearest"`
- **Follow last seen person**: `"last_seen"` or `"me"`
- **Stop following**: `"stop"` or `"stop following"`

## Configuration

### ROS2 Configuration

Add to configuration file:

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
        min_following_distance: 0.8,
        linear_speed_max: 0.5,
        angular_speed_max: 0.5,
        position_tolerance: 0.2,
        angle_tolerance: 0.1,
      },
    },
  ],
}
```

### Zenoh Configuration

```json5
{
  URID: "unitree_go2_autonomy_advance",  // Configure at top level
  agent_actions: [
    {
      name: "follow_person",
      llm_label: "follow_person",
      implementation: "passthrough",
      connector: "zenoh",
      config: {
        URID: "unitree_go2_autonomy_advance",
        person_detection_topic: "person_detection",
        movement_command_topic: "c3/cmd_vel",
        update_rate_hz: 10.0,
        max_following_distance: 5.0,
        min_following_distance: 0.8,
        linear_speed_max: 0.5,
        angular_speed_max: 0.5,
        position_tolerance: 0.2,
        angle_tolerance: 0.1,
      },
    },
  ],
}
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `person_detection_topic` | str | `/person_detection` | Person detection topic |
| `movement_command_topic` | str | `/cmd_vel` | Movement command topic |
| `update_rate_hz` | float | 10.0 | Control loop update rate |
| `max_following_distance` | float | 5.0 | Maximum following distance (meters) |
| `min_following_distance` | float | 0.8 | Minimum safe distance (meters) |
| `linear_speed_max` | float | 0.5 | Maximum linear velocity (m/s) |
| `angular_speed_max` | float | 0.5 | Maximum angular velocity (rad/s) |
| `position_tolerance` | float | 0.2 | Position tolerance (meters) |
| `angle_tolerance` | float | 0.1 | Angle tolerance (radians) |

## Usage Examples

### User Command Examples

1. **Follow by name**
   ```
   User: "Follow Alice"
   LLM: FollowPerson(action="alice", distance=1.5, speed=0.5)
   ```

2. **Follow nearest person**
   ```
   User: "Follow the nearest person"
   LLM: FollowPerson(action="nearest", distance=1.5)
   ```

3. **Follow me**
   ```
   User: "Follow me"
   LLM: FollowPerson(action="last_seen", distance=1.5)
   ```

4. **Follow with specified distance**
   ```
   User: "Follow Bob at 2 meters"
   LLM: FollowPerson(action="bob", distance=2.0, speed=0.6)
   ```

5. **Stop following**
   ```
   User: "Stop following"
   LLM: FollowPerson(action="stop")
   ```

### System Prompt Example

Add to `system_prompt_base` in configuration file:

```json5
system_prompt_base: "You are a helpful robot assistant that can follow people. When asked to follow someone, use the follow_person action. You can follow people by name, follow the nearest person, or follow the last person you saw. Always maintain a safe distance and stop if the person is lost."
```

## Status Feedback

The action provides real-time status information through the `FollowPersonStatus` input, which can be read in the fuser:

- `"following mode=by_name person=alice distance=1.5m"` - Currently following
- `"moving linear=0.3 angular=0.1"` - Currently moving
- `"at target distance=1.5m"` - Reached target distance
- `"person not detected"` - Person not detected
- `"person lost stopping"` - Person lost, stopping follow
- `"timeout stopping"` - Timeout, stopping
- `"stopped"` - Stopped

## Technical Implementation Details

### Person Detection Integration

The action supports obtaining person position information from the following sources:

1. **VLM Input Parsing**: Extract person position from vision language model descriptions
   - Parse pattern: `"person named alice, 2.5 meters away"`
   - Parse pattern: `"alice is 2.5 meters away"`

2. **ROS2 Topic Subscription** (requires implementation)
   - Subscribe to `/person_detection` topic
   - Receive person detection messages

3. **Zenoh Topic Subscription** (requires implementation)
   - Subscribe to `{URID}/person_detection` topic
   - Receive person detection messages

4. **Face Recognition Service** (extensible)
   - Integrate with FacePresenceProvider
   - Get identity information for registered persons

### Control Algorithm

Uses a simplified PID control algorithm:

```python
# Distance control
distance_error = current_distance - target_distance
linear_vel = kp_distance * distance_error * speed_multiplier

# Angle control
angle_error = current_angle
angular_vel = kp_angle * angle_error * speed_multiplier
```

### Safety Mechanisms

1. **Distance Limits**:
   - If person distance > `max_following_distance`: Stop movement
   - If person distance < `min_following_distance`: Move backward

2. **Timeout Protection**:
   - If person is lost for more than 5 seconds: Stop following
   - If total following time exceeds `timeout_sec`: Stop following

3. **Speed Limits**:
   - Both linear and angular velocities are limited to configured maximum values

## Extension Development

### Adding New Person Detection Sources

1. Add new detection method in connector
2. Integrate new source in `_get_person_position()`
3. Ensure return format is `(distance, angle)` tuple

### Improving Control Algorithm

1. Implement complete PID controller
2. Add feedforward control
3. Implement adaptive speed adjustment

### Adding New Features

1. **Multi-person Tracking**: Track multiple people simultaneously
2. **Path Planning**: Use navigation stack for path planning
3. **Gesture Recognition**: Recognize stop, accelerate gestures
4. **Voice Feedback**: Provide voice feedback during following

## Testing Recommendations

### Unit Tests

1. Test interface definition
2. Test control algorithm calculations
3. Test state management

### Integration Tests

1. Test integration with VLM inputs
2. Test ROS2/Zenoh communication
3. Test following behavior

### Scenario Tests

1. **Normal Following**: Person moves normally, robot follows
2. **Person Lost**: Person leaves view, robot stops
3. **Distance Control**: Test different distance settings
4. **Speed Control**: Test different speed settings
5. **Multi-person Scenario**: Test selection logic in multi-person environment

## Troubleshooting

### Issue: Person Not Detected

**Possible Causes**:
- VLM input not configured correctly
- Person detection service not running
- Person not in view

**Solutions**:
- Check VLM input configuration
- Verify person detection service status
- Check if camera is working properly

### Issue: Unstable Following

**Possible Causes**:
- Control parameters set incorrectly
- Update rate too low
- Person detection delay

**Solutions**:
- Adjust `kp_distance` and `kp_angle` parameters
- Increase `update_rate_hz`
- Optimize person detection delay

### Issue: Robot Not Moving

**Possible Causes**:
- ROS2/Zenoh connection failed
- Movement command topic misconfigured
- Permission issues

**Solutions**:
- Check ROS2/Zenoh connection
- Verify topic name configuration
- Check robot permissions

## Complete Configuration File Examples

Refer to the following configuration files:
- `config/follow_person_example.json5` - ROS2 configuration example
- `config/follow_person_zenoh_example.json5` - Zenoh configuration example

## Related Documentation

- [OM1 Architecture Documentation](docs/developing/2_architecture.mdx)
- [Actions Development Guide](docs/developing/6_actions.mdx)
- [Project Structure](docs/developing/7_project_structure.mdx)

## Contributing

We welcome issues and pull requests to improve this action!

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-11  
**Maintainer**: OM1 Team

