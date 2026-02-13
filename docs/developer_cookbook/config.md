---
title: Build a fresh config file
description: "Config"
icon: gear
---

The config file defines the agent that runs on your machine.
It tells OM1 which modules to load, how the robot should behave, and which modes are available.

To ensure your configuration is valid, follow the format defined [here](https://github.com/OpenMind/OM1/tree/main/config/schema).

#### Steps to build a new config file

1. Start with getting your API key from [OpenMind Portal](https://portal.openmind.org/). Copy it and save it, you'll paste it into the config later.
2. Create a new config file config.json5

| Field                    | Type     | Required | Description                                                                      |
| ------------------------ | -------- | -------- | -------------------------------------------------------------------------------- |
| `version`                | `string` | Yes      | The version of the configuration format. Example: `"v1.0.0"`                     |
| `hertz`                  | `number` | Yes      | How often (in Hz) the agent runs its update loop. Example: `0.01`                |
| `name`                   | `string` | Yes      | The name of the agent. Example: `"conversation"`                                 |
| `default_mode`           | `string` | Yes      | The default_mode defines the mode robot starts in. Example: `"welcome"`          |
| `allow_manual_switching` | `bool`   | Yes      | Defines if manual mode switching is allowed. Example: `true`                     |
| `mode_memory_enabled`    | `bool`   | Yes      | Enables or disables mode memory. Example: `true`                                 |
| `api_key`                | `string` | Yes      | API key used to authenticate the agent. Example: `"openmind_free"`               |
| `system_prompt_base`     | `string` | Yes      | Defines the agent's core personality and behavior. Serves as the primary system prompt for the LLM. |
| `system_governance`      | `string` | Yes      | The laws or constraints that the agent must follow during operation. Modeled similarly to Asimov's laws. |
| `system_prompt_examples` | `string` | No       | Example interactions that help guide the model's behavior.                       |

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
| `type`   | `string` | Yes      | The input type identifier. Example: `"AudioInput"`                 |
| `config` | `object` | No       | Configuration options specific to this input type. Example: `GoogleASRInput` |

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
| `implementation` | `string` | No       | Defines the business logic. If none defined, defaults to `"passthrough"`. Example: `"passthrough"`                         |
| `connector`      | `string` | Yes      | Name of the connector. This is the Python file name defined under `actions/action_name/connector`. Example: `"elevenlabs_tts"` |

### Step 7: Add modes

Add `modes` section in your config file and introduce the modes you'd like to configure for you agent.

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

For a better understanding of how modes are configured, refer the documentation [here](new_mode.md)

### Step 8. Validate the config

    Before using the file: Check for JSON errors, make sure commas, quotes, and braces are correct and confirm that correct API key is configured.
