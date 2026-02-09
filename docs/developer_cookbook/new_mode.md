---
title: New Mode
description: "Add a new mode"
icon: gamepad-modern
---

This guide walks you through creating a new mode for your robot system.

## Project Structure

```bash
src/runtime/multi_mode/
├── config.py          # ModeConfig and ModeSystemConfig classes
├── manager.py         # ModeManager for transitions
├── cortex.py          # ModeCortexRuntime for execution
└── hook.py            # Lifecycle hooks

src/runtime/single_mode/
├── config.py          # ModeConfig and ModeSystemConfig classes
├── cortex.py           # ModeCortexRuntime for Execution

config/
└── your_robot_modes.json5    # Mode configuration file
```

## Configuration

### Step 1: Create Configuration File

Create or modify a configuration file (e.g., `your_robot_modes.json5`) in the `/config/` directory.

### Step 2: Add Mode Definition

Add your new mode to the `modes` section of your configuration file.

| Field                  | Type      | Required | Description                                                                                                   |
| ---------------------- | --------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `display_name`         | `string`  | Yes      | The human-readable name shown in the UI for this mode. Example: `"Your New Mode"`                             |
| `description`          | `string`  | Yes      | Brief description explaining what this mode does and its purpose.                                             |
| `system_prompt_base`   | `string`  | Yes      | The foundational system prompt that defines the agent's behavior and purpose in this mode.                    |
| `hertz`                | `float`   | Yes      | The frequency (in Hz) at which the agent operates or processes information. Example: `1.0`                    |
| `timeout_seconds`      | `integer` | Yes      | Maximum duration (in seconds) before the agent times out during execution. Example: `300`                     |
| `remember_locations`   | `boolean` | Yes      | Whether the agent should persist and recall location data across interactions. Example: `false`               |
| `save_interactions`    | `boolean` | Yes      | Whether to save conversation history and interactions for this mode. Example: `true`                          |
| `agent_inputs`         | `array`   | Yes      | List of input sources or data types the agent can accept in this mode.                                        |
| `agent_actions`        | `array`   | Yes      | List of actions or capabilities the agent can perform in this mode.                                           |
| `lifecycle_hooks`      | `array`   | Yes      | Event handlers triggered at specific points in the agent's lifecycle (startup, shutdown, etc.).               |
| `simulators`           | `array`   | Yes      | List of simulation environments or tools available to the agent in this mode.                                 |
| `cortex_llm`           | `object`  | Yes      | Configuration object for the language model powering the agent's cortex.                                      |

### Step 3: Configure Input Plugins

Specify which inputs your mode needs:

| Field    | Type     | Required | Description                                        |
| -------- | -------- | -------- | -------------------------------------------------- |
| `type`   | `string` | Yes      | The input type identifier. Example: `"AudioInput"` |
| `config` | `object` | No       | Configuration options specific to this input type. |

### Step 4: Configure LLM (Optional - Can be overwritten for each mode)

Define which LLM needs to be configured:

| Field            | Type      | Required | Description                                                  |
| ---------------- | --------- | -------- | ------------------------------------------------------------ |
| `type`           | `string`  | Yes      | The LLM provider name. Example: `"OpenAILLM"`                |
| `config`         | `object`  | No       | Configuration options specific to this LLM type.             |
| `agent_name`     | `string`  | No       | Agent name used in metadata. Example: `"Spot"`               |
| `history_length` | `integer` | No       | Number of past messages to remember in the conversation. Example: `10` |

### Step 5: Configure actions

Actions define what your agent can do. You can define movement, TTS or any other actions here.

| Field                 | Type      | Required | Description                                                                                                          |
| --------------------- | --------- | -------- | -------------------------------------------------------------------------------------------------------------------- |
| `name`                | `string`  | Yes      | Human-readable identifier for the action. Example: `"speak"`                                                         |
| `llm_label`           | `string`  | Yes      | Label the model uses to refer to this action. Example: `"speak"`                                                     |
| `implementation`      | `string`  | No       | Defines the business logic. If none defined, defaults to `"passthrough"`. Example: `"passthrough"`                   |
| `connector`           | `string`  | Yes      | Name of the connector. This is the Python file name defined under `actions/action_name/connector`. Example: `"elevenlabs_tts"` |
| `config`              | `object`  | No       | Configuration options specific to this action.                                                                       |
| `exclude_from_prompt` | `boolean` | No       | Whether to exclude this action from the LLM prompt. Default: `false`                                                 |

### Step 6: Add Lifecycle hooks (Only required for multi-mode)

A hook is a programmable event point that executes specific actions at key stages. To define lifecycle hooks, you can add the following to your config.

| Field             | Type      | Required | Description                                                                                                     |
| ----------------- | --------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| `hook_type`       | `string`  | Yes      | The lifecycle event type. Allowed values: `"on_startup"`, `"on_shutdown"`, `"on_entry"`, `"on_exit"`, `"on_timeout"` |
| `handler_type`    | `string`  | Yes      | The type of handler to execute. Allowed values: `"message"`, `"command"`, `"function"`, `"action"`              |
| `handler_config`  | `object`  | Yes      | Configuration for the handler containing one of: `message`, `command`, `function`, or `action` as a string property |
| `priority`        | `integer` | No       | Execution priority when multiple hooks exist for the same event.                                                |
| `async_execution` | `boolean` | No       | Whether to execute the handler asynchronously.                                                                  |
| `timeout_seconds` | `number`  | No       | Maximum duration for handler execution.                                                                         |
| `on_failure`      | `string`  | No       | Behavior when handler fails. Allowed values: `"log"`, `"ignore"`, `"abort"`                                     |

### Step 7: Add Transition Rules (Only required for multi-mode)

Add to the transition_rules section:

| Field              | Type      | Required | Description                                                              |
| ------------------ | --------- | -------- | ------------------------------------------------------------------------ |
| `from_mode`        | `string`  | Yes      | The source mode from which the transition originates.                    |
| `to_mode`          | `string`  | Yes      | The target mode to which the agent will transition.                      |
| `transition_type`  | `string`  | Yes      | The type of transition mechanism.                                        |
| `trigger_keywords` | `array`   | Yes      | List of keywords that trigger this transition. Each item is a string.    |
| `priority`         | `integer` | Yes      | Priority level for this transition rule when multiple rules match.       |
| `cooldown_seconds` | `number`  | No       | Minimum time (in seconds) before this transition can be triggered again. |

### Step 8: Update Default Mode (Optional)

If "new_mode" should be the starting mode, define

    ```bash
    "default_mode": "new_mode"
    ```

Now, your new mode is ready to be tested. Deploy it directly on your robot or configure it through the docker_compose file!
