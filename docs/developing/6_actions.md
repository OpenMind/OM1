---
title: Actions
description: "Actions"
icon: location-arrow
---

## Action Plugins

The Action Plugins are core components of OM1. These plugins map high-level decisions from one or more LLMs into concrete physical or digital actions (e.g. moving a robot or generating speech). This page covers the architecture of a typical Action Plugin, the available action types, and how actions are connected to different hardware and software platforms.

[**Code**](https://github.com/OpenMind/OM1/tree/main/plugins/actions)


## Action Orchestrator

The Action Orchestrator (`internal/actions/orchestrator.go`) manages the execution of the actions selected in each tick. It supports three execution modes — `concurrent` (default), `sequential`, and `dependencies` — configured via the top-level `action_execution_mode` field.

[**Code**](https://github.com/OpenMind/OM1/blob/main/internal/actions/orchestrator.go)

## Movement (Zenoh)

Movement commands can be sent over Zenoh. See the Unitree Go2 autonomy connector, which publishes `cmd_vel` over Zenoh: [`plugins/actions/unitree/go2/autonomy/move.go`](https://github.com/OpenMind/OM1/blob/main/plugins/actions/unitree/go2/autonomy/move.go) (registered as `unitree_go2_autonomy/move`).

## Navigation

The navigation action drives the robot to mapped locations: [`plugins/actions/navigation/navigation.go`](https://github.com/OpenMind/OM1/blob/main/plugins/actions/navigation/navigation.go) (registered as `navigation/navigation`).


## Speech and TTS

The Speech and TTS action plugin allows agents to speak using a text-to-speech (TTS) system.

[**Code**](https://github.com/OpenMind/OM1/blob/main/plugins/actions/speak/elevenlabs_tts.go)


## Adding New Actions

An action is exposed to the LLM by its `name`, and each `name` can have one or more **connectors** that carry it out on a specific platform. A connector registers itself with `actions.Register("<name>/<connector>", ...)` and the runtime resolves it by `name + "/" + connector` (`internal/actions/action.go`).

The real layout groups connectors by action name (and, for robots, by platform):

```tree
plugins/actions/
├── actions.go                                  # blank-imports every action package so init()/Register runs
├── speak/
│   ├── elevenlabs_tts.go                        # speak/elevenlabs_tts
│   ├── elevenlabs_people_tts.go                 # speak/elevenlabs_people_tts
│   └── kokoro_tts.go                            # speak/kokoro_tts
├── emotion/
│   └── zenoh.go                                 # emotion/zenoh
├── navigation/
│   └── navigation.go                            # navigation/navigation
├── face_memory/
│   └── face_memory.go                           # face_memory/face_memory
├── greeting_conversation/
│   └── greeting_conversation_elevenlabs.go      # greeting_conversation/greeting_conversation_elevenlabs
├── robot_action/
│   └── http.go                                  # robot_action/http
└── unitree/
    ├── g1/arm/zenoh.go                           # unitree_g1_arm/zenoh
    └── go2/
        ├── autonomy/move.go                      # unitree_go2_autonomy/move
        ├── autonomy/mppi.go                      # unitree_go2_autonomy/mppi
        └── location/location.go                  # unitree_go2_location/location
```

In general, each robot has specific capabilities, so an action's connector is typically hardware-specific. To add a new connector, create a package under `plugins/actions/`, call `actions.Register("<name>/<connector>", <constructor>)` in its `init()`, and add a blank import for it in `plugins/actions/actions.go` so it is loaded. Then reference it from the `agent_actions` section of your config (e.g. `unitree_g1_conversation.json5`).
