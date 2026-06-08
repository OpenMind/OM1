---
title: Actions
description: "Actions"
icon: location-arrow
---

## Action Plugins

The Action Plugins are core components of OM1. These plugins map high-level decisions from one or more LLMs into concrete physical or digital actions (e.g. moving a robot or generating speech). This page covers the architecture of a typical Action Plugin, the available action types, and how actions are connected to different hardware and software platforms.

[**Code**](https://github.com/OpenMind/OM1/tree/main/plugins/actions)


## Action Orchestrator

The Action Orchestrator is the central component that orchestrates the execution of actions. It manages the states, promise queue, and threads for each action.

[**Code**](https://github.com/OpenMind/OM1/blob/main/internal/actions/orchestrator.go)

## Movement (Zenoh)

This plugin is an example of how to use Zenoh to send movement commands to a [TurtleBot 4](https://github.com/OpenMind/OM1/tree/main/plugins/actions/move_turtle/zenoh.go).

## Movement (Unitree SDK)

This plugin is an example of how to connect to the Unitree SDK to send movement commands to a [Go2 EDU](https://github.com/OpenMind/OM1/tree/main/plugins/actions/move_go2_autonomy/unitree_rplidar_sdk.go).


## Speech and TTS

The Speech and TTS action plugin allows agents to speak using a text-to-speech (TTS) system.

[**Code**](https://github.com/OpenMind/OM1/blob/main/plugins/actions/speak/elevenlabs_tts.go)


## Adding New Actions

Each action plugin consists of:

1. **Interface**: Defines input/output types via Go structs
2. **Implementation**: Business logic (or passthrough for simple actions)
3. **Connector**: Code that connects OM1 to specific virtual or physical environments

```tree
plugins/actions/
├── move_{unique_hardware_id}/
│   ├── interface.go      # Defines MoveInput/Output structs
│   ├── passthrough.go    # Simple passthrough implementation
│   ├── ros2.go           # Maps OM1 data/commands to ROS2
│   ├── zenoh.go          # Maps OM1 data/commands to Zenoh
│   └── unitree.go        # Maps OM1 data/commands to Unitree SDK
└── speak/
    └── elevenlabs_tts.go
```

In general, each robot will have specific capabilities, and therefore, each action will be hardware specific. For example, if you are adding support for the Unitree G1 Humanoid version 13.2b, which supports a new movement subtype such as `dance_2`, you could name the updated action `move_unitree_g1_13_2b` and select that action in your `unitree_g1.json` configuration file.
