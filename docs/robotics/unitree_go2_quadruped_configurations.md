---
title: Unitree Go2 Quadruped
description: "Unitree Go2 EDU Quadruped Configurations"
icon: robot
---

## Manual Movement Control

Regardless of all other settings, you can manually control the dog's movements with an Xbox (or other) controller. Press:

* A to stand up
* B to sit down
* The D-pad allows you to steer the quadruped.
* The front triggers allow you to turn left and right.

Note that game controller has command precedence over the AI, so game controller inputs will override AI-generated physical actions.

## CONFIGURATION 1: Minimal Quadruped Functionality

In this configuration, the quadruped observes its environment, listens and speaks, but there is no AI-controlled movement.

Run
```bash
uv run src/run.py unitree_go2_basic
```

In this mode, the quadruped is configured to (1) use a small local VLM, (2) listen to you, and (3) to speak to you. The amount of speech ("always") is set via the `"silence_rate": 0, // vocalize all speech outputs` setting in `actions:speak:config`.

## CONFIGURATION 2: Full Autonomy

Run
```bash
uv run src/run.py unitree_go2_autonomy
```

OM1 will provide LIDAR and other data to a system of LLMs, allowing them to autonomously explore indoor and outdoor environments.

In this mode, the quadruped is configured to (1) use a cloud VLM, (2) listen to you, and (3) to speak to you occasionally, unless you spoke first, in which case it will always respond. The amount of speech ("sometimes") is set via the `"silence_rate": 6, // vocalize every 6th speech output` setting in `actions:speak:config`.

## CONFIGURATION 3: Autonomous Mapping

Run
```bash
uv run src/run.py unitree_go2_mapper
```

OM1 will provide LIDAR and other data to a system of LLMs, allowing them to autonomously explore indoor and outdoor environments. Also, the system will log position (local odometry and GPS data) and Bluetooth data, as the basis for reliable navigation and path planning. In this mode, the quadruped is configured to use a cloud VLM. There is no speech in this configuration.
