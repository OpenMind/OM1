---
title: Unitree Go2
description: "Unitree Go2 Autonomous Movement Logic"
---

## Hardware needs

The autonomous exploration capability requires a laserscan sensor to be mounted to the head of the Go2. We recommend the [RPLidar A1M8](https://bucket-download.slamtec.com/d1e428e7efbdcd65a8ea111061794fb8d4ccd3a0/LD108_SLAMTEC_rplidar_datasheet_A1M8_v3.0_en.pdf). Please see the [RPLidar setup documentation](motion_planning_lidarA1M8.md) for more information.

## Overview

Using OM1, the Unitree Go2 is able to autonomously explore spaces such as your home. There are several parts to this capability. To get started, launch OM1:

```bash Run OM1
uv run src/run.py unitree_go2_autonomy
```

### RPLIDAR Laserscan Data

OM1 uses the RPLIDAR to tell the core LLMs about nearby objects. This information flows as natural language to the core LLMs. The LIDAR data are also used in the action driver to check for viable paths before and during motions are executed.

### Core LLM Directed Motion

Depending on the sensed spatial environment, the core LLMs can generate contextually appropriate motion commands.

```py
# actions/move_safe_lidar/interface.py
  TURN_LEFT = "turn left"
  TURN_RIGHT = "turn right"
  MOVE_FORWARDS = "move forwards"
  STAND_STILL = "stand still"
```

## Data Priorities

### Normal Case

* The LIDAR does not sense anything in proximity (within 1.1m or closer).

In this case, the Go2 moves about the room controlled by the core LLMs.

### Object Nearby - Possible Moves are Constrained

The LIDAR senses something within 1.1 m (or less) and uses that information to tell the core LLMs about which paths are possible. For example, the LIDAR may tell the core LLMS that:

```bash
Here is information about objects and walls around you. Use this information to plan your movements and avoid bumping into things: The safe movement choices are: You can turn left. You can turn right.
```

If all directions are blocked, the LIDAR tells the LLMs that:

```bash
You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE.
```

In this case, the core LLMs **should** command the Go2 to avoid the object(s).
