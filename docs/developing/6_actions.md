---
title: Actions
description: "Actions"
icon: location-arrow
---

## Action Plugins

The Action Plugins are core components of OM1. These plugins map high-level decisions from one or more LLMs into concrete physical or digital actions (e.g. moving a robot or generating speech). This page covers the architecture of a typical Action Plugin, the available action types, and how actions are connected to different hardware and software platforms.

[**Code**](https://github.com/OpenMind/OM1/tree/main/src/actions)


## Action Orchestrator

The Action Orchestrator is the central component that orchestrates the execution of actions. It manages the states, promise queue, and threads for each action.

[**Code**](https://github.com/OpenMind/OM1/blob/main/src/actions/orchestrator.py)

## Movement (Zenoh)

This plugin is an example of how to use Zenoh to send movement commands to a [TurtleBot 4](https://github.com/OpenMind/OM1/tree/main/src/actions/move_turtle/connector/zenoh.py).

## Movement (Unitree SDK)

This plugin is an example of how to directly connect to the Unitree python SDK to send movement commands to a [Go2 EDU](https://github.com/OpenMind/OM1/tree/main/src/actions/move_go2_autonomy/connector/unitree_rplidar_sdk.py).


## Speech and TTS

The Speech and TTS action plugin allows agents to speak using a text-to-speech (TTS) system.

**Important:** The `speak` action must be explicitly included in your `agent_actions` configuration for the agent to produce speech output. If `speak` is not listed, the LLM may still generate text responses, but they will not be converted to audio.

The following TTS connectors are available:

| Connector | Description |
|-----------|-------------|
| `elevenlabs_tts` | Cloud-based TTS via ElevenLabs. Supports `voice_id` and `silence_rate` config options. |
| `kokoro_tts` | Local TTS using Kokoro. |
| `riva_tts` | NVIDIA Riva TTS. |
| `ub_tts` | UbTech robot TTS. |
| `zenoh` | TTS over Zenoh middleware. |
| `ros2` | TTS over ROS2. |

[**Code**](https://github.com/OpenMind/OM1/blob/main/src/actions/speak/connector/elevenlabs_tts.py)

## Emotion / Face

The Emotion action allows agents to express emotions through facial expressions or LED indicators. Available emotions: `happy`, `sad`, `mad`, `curious`.

| Connector | Description |
|-----------|-------------|
| `avatar` | Web-based avatar facial expressions (used with WebSim). |
| `ros2` | Emotion display over ROS2. |
| `unitree_sdk` | Unitree robot LED color mapping. |

[**Code**](https://github.com/OpenMind/OM1/tree/main/src/actions/emotion)

## Combining speak, move, and emotion

Actions can be executed together so the agent can speak, move, and show emotion simultaneously. To enable this, list all desired actions in `agent_actions` and set `action_execution_mode` to `"concurrent"`:

```json5
{
  action_execution_mode: "concurrent",
  agent_actions: [
    {
      name: "speak",
      llm_label: "speak",
      connector: "elevenlabs_tts",
      config: {
        voice_id: "TbMNBJ27fH2U0VgpSNko",
        silence_rate: 20,
      },
    },
    {
      name: "move",
      llm_label: "move",
      implementation: "passthrough",
      connector: "ros2",
    },
    {
      name: "face",
      llm_label: "emotion",
      connector: "avatar",
    },
  ],
}
```

With this configuration, the LLM can produce function calls for all three actions in a single response, and they will be executed at the same time.

> **Note:** Only actions listed in `agent_actions` are available to the LLM. If you want the agent to speak, `speak` must be explicitly included. The same applies to `move` and `emotion`.

## Adding New Actions

Each action consists of:

1. Interface (`interface.py`): Defines input/output types.
2. Implementation (`implementation/`): Business logic, if any. Otherwise, use passthrough.
3. Connector (`connector/`): Code that connects `OM1` to specific virtual or physical environments, typically through middleware (e.g. custom APIs, `ROS2`, `Zenoh`, or `CycloneDDS`)

```tree
actions/
├── move_{unique_hardware_id}/
│   ├── interface.py      # Defines MoveInput/Output
│   ├── implementation/
│   │   └── passthrough.py
│   └── connector/
│       ├── ros2.py       # Maps OM1 data/commands to hardware layers and robot middleware
│       ├── zenoh.py
│       └── unitree.py
└── orchestrator
```

In general, each robot will have specific capabilities, and therefore, each action will be hardware specific. For example, if you are adding support for the Unitree G1 Humanoid version 13.2b, which supports a new movement subtype such as `dance_2`, you could name the updated action `move_unitree_g1_13_2b` and select that action in your `unitree_g1.json` configuration file.
