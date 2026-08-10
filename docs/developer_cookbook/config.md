---
title: Build a fresh config file
description: "Config"
icon: gear
---

The config file defines the agent that runs on your machine.
It tells OM1 which modules to load, how the robot should behave, and which modes are available.

To ensure your configuration is valid, follow the format defined [here](https://github.com/OpenMind/OM1/tree/main/config/schema).

#### Steps to build a new config file

1. Start with getting your API key from [OpenMind Portal](https://portal.openmind.com/). Copy it and save it, you'll paste it into the config later.
2. Create a new config file, e.g. `config/my_agent.json5`.

OM1 supports two config shapes, each with its own required fields (enforced by `config/schema/`):

- **Single-mode** (`single_mode_schema.json`) — one behavior. **Required:** `version`, `hertz`, `name`, `api_key`, `system_prompt_base`, `system_governance`, `system_prompt_examples`, `agent_inputs`, `cortex_llm`, `agent_actions`.
- **Multi-mode** (`multi_mode_schema.json`) — several modes with transitions. **Required at top level:** `version`, `default_mode`, `api_key`, `system_governance`, `cortex_llm`, `modes`. The per-mode required fields are listed in [Step 7](#step-7-add-modes).

The table below describes the top-level fields:

| Field                    | Type     | Required                     | Description                                                                      |
| ------------------------ | -------- | ---------------------------- | -------------------------------------------------------------------------------- |
| `version`                | `string` | Yes (both)                   | The runtime configuration version. Use `"v1.1.0"`.                               |
| `api_key`                | `string` | Yes (both)                   | API key used to authenticate the agent. Example: `"${OM_API_KEY:-openmind_free}"` |
| `system_governance`      | `string` | Yes (both)                   | The laws or constraints the agent must follow. Modeled on Asimov's laws.         |
| `hertz`                  | `number` | Yes (single-mode)            | How often (in Hz) the agent runs its update loop. Example: `0.01`. In multi-mode, `hertz` is set per mode. |
| `name`                   | `string` | Yes (single-mode)            | The name of the agent. Example: `"conversation"`                                 |
| `system_prompt_base`     | `string` | Yes (single-mode)            | The agent's core personality and behavior. In multi-mode, set per mode.          |
| `system_prompt_examples` | `string` | Yes (single-mode)            | Example interactions that guide the model's behavior.                            |
| `agent_inputs`           | `array`  | Yes (single-mode)            | Input sources. In multi-mode, set per mode.                                      |
| `cortex_llm`             | `object` | Yes (both)                   | The LLM configuration (see [Step 5](#step-5-configure-the-llm)).                 |
| `agent_actions`          | `array`  | Yes (single-mode)            | Actions the agent can perform. In multi-mode, set per mode.                      |
| `default_mode`           | `string` | Yes (multi-mode)             | The mode the robot starts in. Example: `"welcome"`                               |
| `modes`                  | `object` | Yes (multi-mode)             | The map of mode definitions (see [Step 7](#step-7-add-modes)).                   |
| `allow_manual_switching` | `bool`   | No (multi-mode)              | Whether manual mode switching is allowed. Example: `true`                        |
| `mode_memory_enabled`    | `bool`   | No (multi-mode)              | Whether mode memory is enabled. Example: `true`                                  |

### Step 3. Customize the system prompts

    There are three key prompt fields:

    - system_prompt_base

        Defines your agent’s personality and behavior.
        You can keep the “Spot the dog” behavior or edit it to match your needs. You can also provide context to the LLM here.

    - system_governance

        Hard-coded rules the agent must follow (Asimovs laws).

    - system_prompt_examples

        Give your model examples of how to respond. These help shape its responses. You can add more examples if needed.

### Step 4. Configure inputs
    Inputs provide the sensory capabilities that allow robots to perceive their environment

| Field    | Type     | Required | Description                                                        |
| -------- | -------- | -------- | ------------------------------------------------------------------ |
| `type`   | `string` | Yes      | A registered input type. Example: `"GoogleASRInput"`. See the full list in [Inputs](../developing/4_inputs.md) / [Configuration](../developing/3_configuration.md#agent-inputs-agent_inputs). |
| `config` | `object` | No       | Options specific to this input type. Example: `GoogleASRInput` accepts `{ rate: 16000, chunk: 1600 }`. |

### Step 5. Configure the LLM

| Field            | Type      | Required | Description                                                          |
| ---------------- | --------- | -------- | -------------------------------------------------------------------- |
| `type`           | `string`  | Yes      | The LLM provider name. Example: `"OpenAILLM"`                        |
| `config`         | `object`  | No       | Configuration options specific to this LLM type.                     |
| `agent_name`     | `string`  | No       | Agent name used in metadata. Example: `"Spot"`                       |
| `history_length` | `integer` | No       | Number of past messages to remember in the conversation. Example: `10` |

### Step 6. Set up agent actions

    Actions define what your agent can do. You can define movement, TTS or any other actions here.

| Field            | Type     | Required | Description                                                                                                                |
| ---------------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------- |
| `name`           | `string` | Yes      | Human-readable identifier for the action. Example: `"speak"`                                                               |
| `llm_label`      | `string` | Yes      | Label the model uses to refer to this action. Example: `"speak"`                                                           |
| `implementation` | `string` | No       | Optional; commonly `"passthrough"`. Example: `"passthrough"`                                                              |
| `connector`      | `string` | Yes      | The connector for this action. The runtime resolves it as `name + "/" + connector` against the registry in `plugins/actions/`. Example: `"elevenlabs_tts"` (→ `speak/elevenlabs_tts`). |

### Step 7: Add modes

Add `modes` section in your config file and introduce the modes you'd like to configure for you agent.

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
| `lifecycle_hooks`      | `array`   | No       | Event handlers triggered on `on_startup` / `on_entry` / `on_exit` / `on_timeout`.                            |
| `timeout_seconds`      | `number`  | No       | Duration (in seconds) after which a `time_based` transition / `on_timeout` hook can fire.                    |

For a better understanding of how modes are configured, refer the documentation [here](new_mode.md)

### Step 8. Validate and run the config

Before using the file: check for JSON5 errors (commas, quotes, and braces), and confirm the correct API key is configured (via `api_key` or the `OM_API_KEY` environment variable).

Then run your config by its filename (without the `.json5` extension):

```bash
make run CONFIG=my_agent
```

Use `make dev CONFIG=my_agent` for verbose debug logging, and `make list-configs` to see all available configs.
