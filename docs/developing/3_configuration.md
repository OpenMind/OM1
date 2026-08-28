---
title: Configuration
description: "Configuration"
icon: gear
---

## Configuration

Agents are configured via JSON5 files in the `/config` directory. The configuration file is used to define the LLM `system prompt`, agent's inputs, LLM configuration, and actions etc.

> **Single-mode vs multi-mode.** A config can either be **single-mode** (top-level `agent_inputs` / `cortex_llm` / `agent_actions` for one behavior) or **multi-mode** (a `modes` map with transitions). Multi-mode is the canonical structure the runtime executes — a single-mode config is folded into one synthesized mode at load (`internal/config/loader.go`). Write single-mode for a single-behavior agent; use multi-mode once you need more than one behavior. The example below is multi-mode.

Here is an example of the configuration file:

```json5
{
  version: "v1.1.0",
  default_mode: "welcome",
  allow_manual_switching: true,
  mode_memory_enabled: true,

  // Global settings
  api_key: "${OM_API_KEY:-openmind_free}",
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
          type: "VLMGemini",
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
          type: "VLMGemini",
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
      mcp_servers: [
        {
          name: "weather",
          transport: "stdio",
          command: "npx",
          args: ["-y", "@h1deya/mcp-server-weather"],
        },
        {
          name: "github",
          transport: "http",
          url: "https://api.githubcopilot.com/mcp/",
          headers: {
            Authorization: "Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}", // pragma: allowlist secret
          },
        },
      ]
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
* **api_key** The API key for the agent. You can get your API key from the [OpenMind Portal](https://portal.openmind.com/).
* **URID** The Universal Robot ID for the robot. Used to join a decentralized machine-to-machine coordination and communication system (FABRIC).
* **system_prompt_base** Defines the agent's personality and behavior.
* **system_governance** The agent's laws and constitution.
* **system_prompt_examples** The agent's example inputs/actions.
* **default_mode** The default mode for the robot to start in (multi-mode configs only).
* **allow_manual_switching** Whether manual switching of mode is allowed (multi-mode configs only).
* **mode_memory_enabled** Whether mode memory is enabled (multi-mode configs only).

### Optional top-level settings

These fields appear in the shipped configs and are worth knowing when integrating OM1:

* **use_tracer** Enables the execution tracer, optionally with a `quality_scorer`. Example: `use_tracer: { enabled: true, quality_scorer: { enabled: true } }`. See [Tracer & Quality Scorer](tracer.md).
* **knowledge_base** Configures RAG retrieval: `{ knowledge_base_name, base_url, min_score, top_k }`. See [Knowledge Base (RAG)](knowledge_base.md).
* **memory** Configures agent memory (`{ enabled, cloud_connection }`).
* **lifecycle_hooks** / **global_lifecycle_hooks** Hooks fired on `on_startup` / `on_entry` / `on_exit` / `on_timeout`, each with a `handler_type` (`action` | `message` | `command`).
* **agent_backgrounds** / **global_backgrounds** Background tasks that run alongside the cortex loop (see [Backgrounds](8_backgrounds.md)).
* **action_execution_mode** *(set per mode, not top-level)* How actions in a tick are executed: `concurrent` (default), `sequential`, or `dependencies` (uses **action_dependencies**). Define this inside a mode block; in a single-mode config it lives alongside that mode's `actions`.
* **robot_ip** IP address of the robot for middleware connections.
* **use_sim** Set `true` when running against a simulator.

## version

The version field specifies the runtime configuration version. It is required for both single-mode and multi-mode configs.

This field ensures that configuration files remain compatible as the runtime evolves. When the version in a config doesn’t match what the runtime expects, developers receive clear logs and errors instead of silent failures or unpredictable behavior.

### Runtime support

The internal/config/version.go module handles:

  - retrieving the current runtime version
  - checking compatibility between config and runtime
  - producing detailed logs and helpful error messages when mismatches occur

### Current version

The current runtime version is **`v1.1.0`** (defined by `LatestRuntimeVersion` in `internal/config/version.go`). Set `version: "v1.1.0"` in every config; all shipped configs in `/config` use this value.

Compatibility is checked at load time (`IsVersionSupported`):

  - **Major version mismatch** (e.g. `v2.x` config on a `v1.x` runtime) → the config is **rejected with an error**.
  - **Minor version mismatch** (e.g. `v1.0` config on a `v1.1` runtime) → the config **loads with a warning** logged.
  - An empty `version` is an error.

> **Note:** Always use the current runtime version in your configuration files unless you have a specific reason to pin an older one.

### Environment variables

Every string value supports shell-style interpolation with defaults: `${VAR:-default}`. This is used throughout the shipped configs, e.g. `api_key: "${OM_API_KEY:-openmind_free}"` and `base_url: "${KB_BASE_URL:-http://localhost:8100}"`. Define the variable in your environment (or a `.env` file) to override the default.

## Agent Inputs (`agent_inputs`)

Example configuration for the agent_inputs section:

```json5
  agent_inputs: [
    {
      type: "GoogleASRInput",
      config: {
        rate: 16000,
        chunk: 1600,
        enable_tts_interrupt: true,
      },
    },
    {
      type: "VLMGemini",
    },
  ]
```

The `agent_inputs` section defines the inputs for the agent. Inputs might include a camera, a microphone, localization, or conversation history. The input `type` must match a registered input plugin. The currently registered input types are:

**Speech (ASR)**
* `GoogleASRInput`, `GoogleASRRTSPInput`
* `ElevenLabsASRInput`, `ElevenLabsASRRTSPInput`
* `RivaASRInput`, `RivaASRRTSPInput`
* `ParallelASRInput`

**Vision (VLM)**
* `VLMGemini`, `VLMGeminiRTSP`
* `VLMOpenAI`, `VLMOpenAIRTSP`
* `VLMBackground`

**Perception & state**
* `FacePresence`
* `LocalizationInput`, `LocationsInput`
* `UnitreeGo2Odom`
* `ConversationHistoryInput`, `GreetingStatus`, `Paths`

> The authoritative list is whatever is registered via `inputs.Register(...)` under `plugins/inputs/`. You can implement your own inputs by following the [Input Plugin Guide](4_inputs.md). Each input's `config` block is specific to its type — e.g. `GoogleASRInput` accepts `rate`, `chunk`, and `enable_tts_interrupt` (see [VAD & TTS Interrupt](vad_tts_interrupt.md)).

## Cortex LLM (`cortex_llm`)

The `cortex_llm` field allows you to configure the Large Language Model (LLM) used by the agent. In a typical deployment, data will flow to at least three different LLMs, hosted in the cloud, that work together to provide actions to your robot.

### Robot Control by a Single LLM

Here is an example configuration of the `cortex_llm` showing use of a single LLM to generate decisions:

```json5
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

## Agent Actions (`agent_actions`)

Defines the agent's available capabilities. Each action has a `name` (exposed to the LLM), an optional `llm_label`, and a `connector` that executes it. Internally the runtime resolves the connector by `name + "/" + connector` (see `internal/actions/action.go`), so the `connector` value must match a registered connector for that action name. Here is an example configuration for the `agent_actions` section:

```json5
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
    {
      name: "emotion",
      llm_label: "emotion",
      connector: "zenoh",
    },
  ]
```

The currently registered action connectors (as `name/connector`) are:

* **speak** — `speak/elevenlabs_tts`, `speak/elevenlabs_people_tts`, `speak/kokoro_tts`
* **emotion** — `emotion/zenoh`
* **navigation** — `navigation/navigation`
* **face_memory** — `face_memory/face_memory`
* **greeting_conversation** — `greeting_conversation/greeting_conversation_elevenlabs`
* **robot_action** — `robot_action/http`
* **Unitree** — `unitree_go2_autonomy/move`, `unitree_go2_autonomy/mppi`, `unitree_go2_location/location`, `unitree_g1_arm/zenoh`

> The authoritative list is whatever is registered via `actions.Register(...)` under `plugins/actions/`. You can add your own actions and connectors following the [Action Plugin Guide](6_actions.md).

## MCP servers

MCP servers can be added to a config to give OM1 agent capability to interact with different MCP tools. Example:

```json5
mcp_servers: [
    {
      name: "weather",
      transport: "stdio",
      command: "npx",
      args: ["-y", "@h1deya/mcp-server-weather"],
    },
  ]
```

Refer to [MCP Integration](mcp-integration.md) to understand the complete architecture and how to configure new MCP tools with OM1.

## Transition rules

Transition rules define how and when the robot switches between operational modes.

```json5
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
