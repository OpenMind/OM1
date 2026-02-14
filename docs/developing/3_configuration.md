---
title: Configuration
description: "Configuration"
icon: gear
---

## Configuration

Agents are configured via JSON5 files in the `/config` directory. The configuration file is used to define the LLM `system prompt`, agent's inputs, LLM configuration, and actions etc. Here is an example of the configuration file:

```python
{
  version: "v1.0.2",
  default_mode: "welcome",
  allow_manual_switching: true,
  mode_memory_enabled: true,

  // Global settings
  api_key: "openmind_free",
  system_governance: "Here are the laws that govern your actions. Do not violate these laws.\nFirst Law: A robot cannot harm a human or allow a human to come to harm.\nSecond Law: A robot must obey orders from humans, unless those orders conflict with the First Law.\nThird Law: A robot must protect itself, as long as that protection doesn't conflict with the First or Second Law.\nThe First Law is considered the most important, taking precedence over the second and third laws.",
  cortex_llm: {
    type: "OpenAILLM",
    config: {
      agent_name: "Bits",
      history_length: 10,
    },
  },

  modes: {
    welcome: {
      display_name: "Welcome Mode",
      description: "Initial greeting and user information gathering",
      system_prompt_base: "You are Bits, a friendly robotic dog meeting someone for the first time. Your goal is to:\n1. Introduce yourself warmly\n2. Ask for the user's name and basic preferences\n3. Explain your capabilities\n4. Ask what they'd like to do together\n\nBe enthusiastic, friendly, and helpful. Keep responses concise but warm.",
      hertz: 0.01,
      agent_inputs: [
        {
          type: "VLM_COCO_Local",
          config: {
            camera_index: 0,
          },
        },
        {
          type: "GoogleASRInput",
        },
      ],
      agent_actions: [
        {
          name: "speak",
          llm_label: "speak",
          connector: "elevenlabs_tts",
          config: {
            voice_id: "TbMNBJ27fH2U0VgpSNko",
            silence_rate: 0,
          },
        },
      ],
    },

    conversation: {
      display_name: "Social Interaction",
      description: "Focused conversation and social interaction mode",
      system_prompt_base: "You are Bits in conversation mode. Focus on:\n1. Engaging in meaningful dialogue\n2. Answering questions thoughtfully\n3. Showing interest in the user\n4. Being a good companion\n5. Responding to emotional cues\n\nBe attentive, empathetic, and engaging. Use appropriate body language and expressions to enhance communication.",
      save_interactions: true,
      hertz: 1,
      agent_inputs: [
        {
          type: "GoogleASRInput",
        },
        {
          type: "VLM_COCO_Local",
          config: {
            camera_index: 0,
          },
        },
      ],
      agent_actions: [
        {
          name: "speak",
          llm_label: "speak",
          connector: "elevenlabs_tts",
          config: {
            voice_id: "TbMNBJ27fH2U0VgpSNko",
            silence_rate: 10,
          },
        },
      ],
    },
  },

  transition_rules: [
    // From welcome mode
    {
      from_mode: "welcome",
      to_mode: "conversation",
      transition_type: "input_triggered",
      trigger_keywords: [
        "talk",
        "chat",
        "conversation",
        "tell me",
        "ask you",
        "discuss",
      ],
      priority: 2,
      cooldown_seconds: 3.0,
    },

    // Universal transitions (from any mode)
    {
      from_mode: "*",
      to_mode: "welcome",
      transition_type: "input_triggered",
      trigger_keywords: [
        "reset",
        "start over",
        "welcome mode",
        "restart",
        "initialize",
      ],
      priority: 5,
      cooldown_seconds: 10.0,
    },
  ],
}
```

## Common Configuration Elements

* **hertz** Defines the base tick rate of the agent. This rate can be adjusted to allow the agent to respond quickly to changing environments, but comes at the expense of reducing the time available for LLMs to finish generating tokens. Note: time critical tasks such as collision avoidance should be handled through low level control loops operating in parallel to the LLM-based logic, using event-triggered callbacks through real-time middleware.
* **name** A unique identifier for the agent.
* **api_key** The API key for the agent. You can get your API key from the [OpenMind Portal](https://portal.openmind.org/).
* **URID** The Universal Robot ID for the robot. Used to join a decentralized machine-to-machine coordination and communication system (FABRIC).
* **system_prompt_base** Defines the agent's personality and behavior.
* **system_governance** The agent's laws and constitution.
* **system_prompt_examples** The agent's example inputs/actions.
* **default_mode** The default mode for the robot to start in.
* **allow_manual_switching** To decide if manual switching of mode is allowed or not.
* **mode_memory_enabled** Whether mode memory is enabled.

## version

The version field specifies the runtime configuration version. It is required for both single-mode and multi-mode configs.

This field ensures that configuration files remain compatible as the runtime evolves. When the version in a config doesn’t match what the runtime expects, developers receive clear logs and errors instead of silent failures or unpredictable behavior.

### Runtime support

The runtime/version.py module handles:

  - retrieving the current runtime version
  - checking compatibility between config and runtime
  - producing detailed logs and helpful error messages when mismatches occur

### Available versions

  - `v1.0.0`

    Initial stable configuration version.

  - `v1.0.1`

    Adds support for context-aware mode for full autonomy.

  - `v1.0.2` (latest)

    Adds support for multiple TTS.

> **Note:** Always use the latest supported version in your configuration files unless you have a specific reason to pin an older version.

## Agent Inputs (`agent_inputs`)

Example configuration for the agent_inputs section:

```python
  agent_inputs: [
    {
      type: "GovernanceEthereum"
    },
    {
      type: "VLM_COCO_Local",
      config: {
        camera_index: 0
      }
    }
  ]
```

The `agent_inputs` section defines the inputs for the agent. Inputs might include a camera, a LiDAR, a microphone, or governance information. OM1 implements the following input types:

* GoogleASRInput
* VLMVila
* VLM_COCO_Local
* RPLidar
* TurtleBot4Batt
* UnitreeG1Basic
* UnitreeGo2Lowstate
* GovernanceEthereum
* more being added continuously...

You can implement your own inputs by following the [Input Plugin Guide](4_inputs.md). The `agent_inputs` config section is specific to each input type. For example, the `VLM_COCO_Local` input accepts a `camera_index` parameter.

## Cortex LLM (`cortex_llm`)

The `cortex_llm` field allows you to configure the Large Language Model (LLM) used by the agent. In a typical deployment, data will flow to at least three different LLMs, hosted in the cloud, that work together to provide actions to your robot.

### Robot Control by a Single LLM

Here is an example configuration of the `cortex_llm` showing use of a single LLM to generate decisions:

```python
  cortex_llm: {
    type: "OpenAILLM",
    config: {
      base_url: "",       // Optional: URL of the LLM endpoint
      api_key: "...",     // Optional: Override the default API key
      agent_name: "Iris", // Optional: Name of the agent
      history_length: 10
    }
  }
```

* **type**: Specifies the LLM plugin.
* **config**: LLM configuration, including the API endpoint (`base_url`), `agent_name`, and `history_length`.

You can directly access other OpenAI style endpoints by specifying a custom API endpoint in your configuration file. To do this, provide a suitable `base_url` and the `api_key` for OpenAI, DeepSeek, or other providers. Possible `base_url` choices include:

* https://api.openai.com/v1
* https://api.deepseek.com/v1
* http://localhost:11434 (Ollama - local inference, no API key required)

You can implement your own LLM endpoints or use more sophisticated approaches such as multiLLM robotics-focused endpoints by following the [LLM Guide](5_llms.md).

## Simulators (`simulators`)

Lists the simulation modules used by the agent. Here is an example configuration for the `simulators` section:

```python
  simulators: [
    {
      type: "WebSim",
      config: {
        host: "0.0.0.0",
        port: 8000,
        tick_rate: 100,
        auto_reconnect: true,
        debug_mode: false
      }
    }
  ]
```

## Agent Actions (`agent_actions`)

Defines the agent's available capabilities, including action names, their implementation, and the connector used to execute them. Here is an example configuration for the `agent_actions` section:

```python
  agent_actions: [
    {
      name: "move",
      llm_label: "move",
      implementation: "passthrough",
      connector: "ros2"
    },
    {
      name: "speak",
      llm_label: "speak",
      implementation: "passthrough",
      connector: "ros2"
      config: {
        voice_id: "TbMNBJ27fH2U0VgpSNko",
        silence_rate: 0,
      },
  }
  ]
```

You can customize the actions following the [Action Plugin Guide](6_actions.md)

## Transition rules

Transition rules define how and when the robot switches between operational modes.

```python
    {
      from_mode: "<current_mode>",
      to_mode: "welcome",
      transition_type: "input_triggered",
      trigger_keywords: [
        "reset",
        "start over",
        "welcome mode",
        "restart",
        "initialize",
      ],
      priority: 5,
      cooldown_seconds: 10.0,
    }
```
To understand transition rules in depth, refer the documentation [here](../full_autonomy_guidelines/transition_rules.md)

To introduce a new mode in your config, refer [introduce new mode](../developer_cookbook/new_mode.md)
