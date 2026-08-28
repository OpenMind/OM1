---
title: Person Following
description: "Have the robot follow a person autonomously."
icon: person-walking
---

Person following lets the robot pick out a person and keep pace with them as they move — the basis for guide, companion, and "follow me" behaviors. It ties perception (detecting and tracking the person) to navigation: the robot keeps a following distance and uses the same [obstacle avoidance](obstacle-avoidance.md) as everything else, so it follows safely rather than blindly.

This runs as part of the managed OM1 autonomy stack. You enable it and tune its behavior — things like follow distance and how a target is chosen — through the [OpenMind portal](https://portal.openmind.com) rather than the robot's local API. How well it tracks depends on the platform's sensors and the lighting, and availability depends on your plan.

To turn it on for a deployment, talk to your OpenMind contact. Related: [Navigation](navigation.md) and [Obstacle Avoidance](obstacle-avoidance.md).
