# Mode System for OM1

The OM1 Mode System allows robots to operate in different behavioral modes, each with their own configuration, inputs, actions, and system prompts. This enables complex behaviors like transitioning from a welcoming introduction mode to autonomous exploration or conversational interaction.

## Overview

The mode system consists of three main components:

1. **Mode Configuration** - Defines available modes and transition rules
2. **Mode Manager** - Handles mode transitions and state management
3. **Mode-Aware Cortex** - Runtime that can dynamically switch between modes

## Key Features

- **Dynamic Mode Switching** - Switch between modes automatically or manually
- **Mode-Specific Configurations** - Each mode can have different:
  - System prompts and personality
  - Input sensors (camera, lidar, microphone, etc.)
  - Available actions (movement, speech, etc.)
  - LLM settings (temperature, history length)
  - Update frequency (hertz)
- **Automatic Transitions** - Modes can switch based on:
  - Keywords in user input
  - Time-based triggers (timeouts)
  - Context-aware conditions
- **Transition Management** - Cooldowns, priorities, and graceful switching
- **State Persistence** - Mode history and user context

## Configuration Structure

Mode-aware configurations use a different structure than standard OM1 configs:

```json5
{
  // Global mode system settings
  "default_mode": "welcome",
  "allow_manual_switching": true,
  "transition_announcement": true,
  "mode_memory_enabled": true,

  // Shared settings
  "api_key": "your_api_key",
  "system_governance": "Robot laws and safety rules...",

  // Mode definitions
  "modes": {
    "welcome": {
      "display_name": "Welcome Mode",
      "description": "Initial greeting and user information gathering",
      "system_prompt_base": "You are a friendly robot meeting someone new...",
      "hertz": 2,
      "entry_message": "Hello! I'm Bits, your robotic companion.",
      "agent_inputs": [...],
      "agent_actions": [...]
    },
    "slam": {
      "display_name": "SLAM Exploration",
      "description": "Autonomous navigation and mapping mode",
      "system_prompt_base": "You are in exploration mode...",
      "hertz": 1,
      "remember_locations": true,
      "agent_inputs": [...],
      "agent_actions": [...]
    }
  },

  // Transition rules
  "transition_rules": [
    {
      "from_mode": "welcome",
      "to_mode": "slam",
      "transition_type": "input_triggered",
      "trigger_keywords": ["explore", "map", "navigate"],
      "priority": 3,
      "cooldown_seconds": 5.0
    }
  ]
}
```

## Mode Configuration Options

### Mode Settings

- `display_name` - Human-readable name for the mode
- `description` - Brief description of the mode's purpose
- `system_prompt_base` - The core system prompt for this mode
- `hertz` - Update frequency (how often the cortex processes)
- `entry_message` - Message displayed when entering the mode
- `exit_message` - Message displayed when leaving the mode
- `timeout_seconds` - Auto-exit the mode after this duration
- `remember_locations` - Enable location memory for SLAM modes
- `save_interactions` - Save conversation history

### Mode Components

Each mode can specify its own:
- `agent_inputs` - Input sensors and processors
- `agent_actions` - Available actions
- `cortex_llm` - LLM configuration (or use global)
- `simulators` - Simulation environments
- `backgrounds` - Background processes

## Transition Rules

Transition rules define how and when the robot should switch between modes:

### Transition Types

1. **Input Triggered** - Switch based on keywords in user input
   ```json5
   {
     "transition_type": "input_triggered",
     "trigger_keywords": ["explore", "map", "navigate"],
     "priority": 3,
     "cooldown_seconds": 5.0
   }
   ```

2. **Time Based** - Auto-exit after timeout
   ```json5
   {
     "transition_type": "time_based",
     "priority": 1
   }
   ```

3. **Context Aware** - Switch based on environmental conditions (future)
4. **Manual** - Explicit mode switching commands

### Rule Properties

- `from_mode` - Source mode ("*" for any mode)
- `to_mode` - Target mode
- `priority` - Higher priority rules are checked first
- `cooldown_seconds` - Minimum time between using this rule
- `trigger_keywords` - Words/phrases that activate the transition

## Example Modes

### Welcome Mode
- **Purpose**: Initial greeting, gather user information
- **Inputs**: Camera, microphone
- **Actions**: Speech, basic movements
- **Behavior**: Friendly introduction, ask for user's name and preferences

### SLAM Mode
- **Purpose**: Autonomous exploration and mapping
- **Inputs**: LIDAR, camera, odometry
- **Actions**: Navigation, obstacle avoidance, speech
- **Behavior**: Explore environment, build map, avoid obstacles

### Conversation Mode
- **Purpose**: Social interaction and dialogue
- **Inputs**: Microphone, camera for emotion detection
- **Actions**: Speech, expressive movements
- **Behavior**: Engaging conversation, emotional responses

### Guard Mode
- **Purpose**: Security monitoring and patrol
- **Inputs**: Camera, LIDAR, microphone
- **Actions**: Movement, alert notifications
- **Behavior**: Patrol routes, detect anomalies, security alerts

## Usage

### Starting with Modes

Use the enhanced run script to start mode-aware configurations:

```bash
# Start with mode configuration
uv run src/run_modes.py start unitree_go2_basic_modes

# View available modes
uv run src/run_modes.py modes unitree_go2_basic_modes

# List all configurations
uv run src/run_modes.py list-configs
```

### Runtime Mode Control

During runtime, modes can be switched:

1. **Automatic** - Based on user input keywords
   - "Let's explore" → Switch to SLAM mode
   - "Let's talk" → Switch to conversation mode

2. **Time-based** - Automatic timeout transitions
   - Guard mode auto-exits after 10 minutes

3. **Manual** - Programmatic control through the API

### Monitoring Modes

The system provides information about:
- Current mode and duration
- Available transitions
- Mode history
- Time remaining (for timeout modes)

## Best Practices

### Mode Design

1. **Clear Purpose** - Each mode should have a distinct behavioral purpose
2. **Appropriate Frequency** - Set hertz based on mode requirements:
   - High frequency (2-10 Hz) for real-time interaction
   - Low frequency (0.5-1 Hz) for background monitoring
3. **Resource Management** - Consider computational load of inputs/actions

### Transition Design

1. **Intuitive Keywords** - Use natural language triggers
2. **Appropriate Cooldowns** - Prevent rapid mode switching
3. **Priority Management** - Higher priority for safety/critical transitions
4. **Fallback Modes** - Always provide paths back to safe modes

### System Prompts

1. **Mode-Specific Personality** - Tailor behavior to mode purpose
2. **Context Awareness** - Reference mode capabilities in prompts
3. **Transition Guidance** - Help users understand available modes

## Implementation Notes

- The mode system is backward compatible with existing OM1 configurations
- Mode transitions are graceful - components are stopped and restarted cleanly
- User context and interaction history can be preserved across modes
- The system supports both reactive (input-triggered) and proactive (time-based) transitions

## Future Enhancements

- **Context-Aware Transitions** - Switch based on environmental conditions
- **Learning Transitions** - Adapt transition rules based on usage patterns
- **Hierarchical Modes** - Sub-modes within major behavioral modes
- **Conditional Actions** - Mode-specific action availability
- **Multi-Robot Coordination** - Synchronized mode switching across robot teams
