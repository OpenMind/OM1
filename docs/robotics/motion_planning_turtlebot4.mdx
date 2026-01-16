---
title: TurtleBot4
description: "TurtleBot4 Autonomous Movement Logic"
---

## Overview

Using OM1, the TurtleBot4 (TB4) is able to autonomously explore spaces such as your home. There are several parts to this capability. To get started, launch OM1:

```bash
uv run src/run.py turtlebot4_lidar
```

### TB4 RPLIDAR Laserscan Data

OM1 uses the TB4's RPLIDAR to tell the core LLMs about nearby objects. This information flows to the core LLMs from `/input/plugins/rplidar.py`. The RPLIDAR data are also used in the action driver to check for viable paths right before motions are executed. See the [RPLidar setup documentation](motion_planning_lidarA1M8.mdx) for more information.

### Core LLM Directed Motion

Depending on the environment of the TB4, the core LLMs can generate contextually appropriate motion commands.

```py
# /actions/move_turtle/interface.py
TURN_LEFT = "turn left"
TURN_RIGHT = "turn right"
MOVE_FORWARDS = "move forwards"
STAND_STILL = "stand still"
```
These commands are defined in `actions/move_turtle/interface.py` and are converted to TB4 zenoh/cycloneDDS `cmd_vel` motions in `/actions/move_turtle/connector/zenoh.py`.

### TB4 Physical Collision Switches

In addition to LIDAR data, the TB4 also uses collision switches to detect hazards. When those switches are triggered, two things happen:

1. TB4 Basic Low Level (Firmware) Collision Avoidance

Immediately after a frontal (or side) collision, the TB4 will back off about 10cm. That avoidance motion is handled within the `Create3` and cannot be changed by a user.

2. TB4 Enhanced Collision Avoidance

Beyond the immediate 10cm rewards motion, OM1 uses the TB4's collision switches to invoke an enhanced object avoidance behavior, which consists of turning 100 deg left or right, depending on which switch of several side or frontal collision switches were triggered. This "turning to face away" from the object is handled directly inside the `action` driver to ensure prompt responses to physical collisions:

```py
# /actions/move_turtle/connector/zenoh.py
# this is simplified example code - actual code will differ
def listenerHazard(data):
    global gHazard
    gHazard = sensor_msgs.HazardDetectionVector.deserialize(data.payload.to_bytes())

if gHazard is not None and gHazard.detections and len(gHazard.detections) > 0:
  for haz in gHazard.detections:
      if haz.type == 1:
          if "left" in haz.header.frame_id:
              self.hazard = "TURN_RIGHT"

if self.hazard is not None:
  if self.hazard == "TURN_RIGHT":
      target_yaw = self.yaw_now + 100.0
      if target_yaw >= 180.0: target_yaw -= 360.0
      self.emergency = target_yaw
```

## Object Avoidance and Collision Switch States

### Normal

* The LIDAR does not sense anything in proximity (within 1m or closer).
* The collision switches are open.

In this case, the TB4 moves about the room controlled by the core LLMs.

### Object Nearby and Possible Moves are Constrained

* The LIDAR senses objects in proximity and informs the core LLMs about which paths are possible.
* The collision switches are open.

In this case, the core LLMs **should** command the TB4 to turn away from the object.

### Collision Switches Triggered

* The collision switches are triggered.

In this case, the firmware logic will command an immediate 10 cm retreat, and then, the `action` level collision avoidance code will command a 100 deg avoidance rotation. Once this rotation is complete, the system reverts to responding to commands from the core LLMs.
