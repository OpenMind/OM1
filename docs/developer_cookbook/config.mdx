---
title: Build a fresh config file
description: "Config"
---

The config file defines the agent that runs on your machine.
It tells OM1 which modules to load, how the robot should behave, and which modes are available.

To ensure your configuration is valid, follow the format defined [here](https://github.com/OpenMind/OM1/tree/main/config/schema).

OpenMind supports two configuration schemas:

### Step 1. Single-Mode Schema

Use this when your robot only needs to run one dedicated mode. For example, a pure conversation agent or a navigation-only setup.
The entire agent is optimized around a single use case.

### Step 2. Multi-Mode Schema

Use this when you want the robot to switch between multiple modes at runtime based on user choice or context.
You can configure any subset of the available five modes, just two, or all five, depending on your application.

#### Steps to build a new config file

1. Start with getting your API key from [OpenMind Portal](https://portal.openmind.org/). Copy it and save it, you'll paste it into the config later.
2. Create a new config file config.json5

| Field                    | Type     | Required | Description                                                                      |
| ------------------------ | -------- | -------- | -------------------------------------------------------------------------------- |
| `version`                | `string` | Yes      | The version of the configuration format. Example: `"v1.0.0"`                     |
| `hertz`                  | `number` | Yes      | How often (in Hz) the agent runs its update loop. Example: `0.01`                |
| `name`                   | `string` | Yes      | The name of the agent. Example: `"conversation"`                                 |
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

### Step 7. Validate the config

    Before using the file: Check for JSON errors, make sure commas, quotes, and braces are correct and confirm that correct API key is configured.
