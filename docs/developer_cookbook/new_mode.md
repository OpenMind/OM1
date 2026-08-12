---
title: New Mode
description: "Add a new mode"
icon: gamepad-modern
---

This guide walks you through creating a new mode for your robot system.

## Project Structure

```bash
internal/config/
└── types.go           # ModeConfig, ModeSystemConfig, TransitionRule, HookSpec types

internal/runtime/
├── config.go          # Mode config wiring + legacy single-mode → multi-mode conversion
├── manager.go         # Mode manager: transitions and lifecycle
└── runtime.go         # Core runtime / cortex loop

config/
└── your_robot_modes.json5    # Mode configuration file
```

> **Note:** Multi-mode is the canonical config structure — it is the only shape the runtime executes. A single-mode config is a convenience shorthand: at load time OM1 folds its top-level fields into one synthesized mode (`internal/config/loader.go`), so you never lose anything by writing single-mode, but multi-mode is preferred for anything beyond a single behavior.

## Configuration

### Step 1: Create Configuration File

Create or modify a configuration file (e.g., `your_robot_modes.json5`) in the `/config/` directory.

### Step 2: Add Mode Definition

Add your new mode to the `modes` section of your configuration file.

Per the schema, each mode **requires** `display_name`, `description`, `system_prompt_base`, `hertz`, `agent_inputs`, and `agent_actions`. Everything else is optional.

| Field                  | Type      | Required | Description                                                                                                   |
| ---------------------- | --------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| `display_name`         | `string`  | Yes      | The human-readable name shown in the UI for this mode. Example: `"Your New Mode"`                             |
| `description`          | `string`  | Yes      | Brief description explaining what this mode does and its purpose.                                             |
| `system_prompt_base`   | `string`  | Yes      | The foundational system prompt that defines the agent's behavior in this mode.                                |
| `hertz`                | `number`  | Yes      | The frequency (in Hz) at which the agent's loop runs in this mode. Example: `1.0`                             |
| `agent_inputs`         | `array`   | Yes      | Input sources the agent accepts in this mode.                                                                 |
| `agent_actions`        | `array`   | Yes      | Actions the agent can perform in this mode.                                                                   |
| `cortex_llm`           | `object`  | No       | Per-mode LLM override. If omitted, the top-level `cortex_llm` is used.                                        |
| `agent_backgrounds`    | `array`   | No       | Background tasks that run while this mode is active.                                                          |
| `mcp_servers`          | `array`   | No       | MCP servers available in this mode.                                                                           |
| `lifecycle_hooks`      | `array`   | No       | Event handlers triggered at specific points in the agent's lifecycle (see [Step 6](#step-6-add-lifecycle-hooks-only-required-for-multi-mode)). |
| `timeout_seconds`      | `number`  | No       | Duration (in seconds) after which a `time_based` transition / `on_timeout` hook can fire.                    |

### Step 3: Configure Input Plugins

Specify which inputs your mode needs:

| Field    | Type     | Required | Description                                        |
| -------- | -------- | -------- | -------------------------------------------------- |
| `type`   | `string` | Yes      | A registered input type. Example: `"GoogleASRInput"`. See [Inputs](../developing/4_inputs.md) for the full list. |
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
| `implementation`      | `string`  | No       | Optional; commonly `"passthrough"`. Example: `"passthrough"`                                                          |
| `connector`           | `string`  | Yes      | The connector for this action; resolved as `name + "/" + connector` against the registry in `plugins/actions/`. Example: `"elevenlabs_tts"` |
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
| `transition_type`  | `string`  | Yes      | The type of transition mechanism: `"input_triggered"`, `"time_based"`, or `"context_aware"`. |
| `priority`         | `integer` | Yes      | Priority level for this transition rule when multiple rules match.       |
| `trigger_keywords` | `array`   | No       | Keywords that trigger this transition (used by `input_triggered`). Not required for `time_based` / `context_aware`. |
| `cooldown_seconds` | `number`  | No       | Minimum time (in seconds) before this transition can be triggered again. |

### Step 8: Update Default Mode (Optional)

If "new_mode" should be the starting mode, set the top-level `default_mode`:

```json5
"default_mode": "new_mode"
```

Now, your new mode is ready to be tested. Deploy it directly on your robot or configure it through the docker_compose file!
