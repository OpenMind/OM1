---
title: Obstacle Avoidance
description: "Avoid obstacles automatically while navigating."
icon: triangle-exclamation
---

A static map only captures the world as it was when you mapped it. Obstacle avoidance handles everything that wasn't there — people walking by, a chair that moved, a box left in the aisle. It's part of the [navigation](navigation.md) stack: as the robot drives toward a goal, it watches its LiDAR and depth sensors and continuously replans the local path around whatever it sees. If there's no safe way through, the goal comes back `ABORTED` rather than the robot forcing an unsafe move.

There's nothing to switch on — it's active automatically whenever the robot is navigating with Nav2. How well it works depends on the platform (some robots offer basic object avoidance) and on sensor coverage: obstacles that sit below the sensors' field of view, or that are small or reflective, may not be caught reliably, so validate it in your actual environment. And treat it as navigation-level smarts, not a safety system — time-critical stops belong to the robot's own low-level safety layer.

In the [OpenMind portal](https://portal.openmind.com) map view, you can watch the planned path bend around obstacles in real time as the robot drives.

If the robot keeps stopping or aborting near clutter, there usually isn't a safe local path — clear the obstacle or start it from a cleaner pose. See [Navigation](navigation.md) for goal handling and the [Autonomy API Overview](../api_endpoints.md) for details.
