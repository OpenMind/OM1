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

The Speech and TTS action plugin allows agents to speak using with a text-to-speech (TTS) system.

[**Code**](https://github.com/OpenMind/OM1/blob/main/src/actions/speak/connector/elevenlabs_tts.py)


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
