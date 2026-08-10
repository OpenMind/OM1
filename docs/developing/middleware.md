---
title: Middleware Setup
description: "ROS2 and DDS Setup"
icon: gear
---

This section focuses on installation guidelines for the ROS 2 middleware stack and related tools.

> **Two different "Zenoh" things — don't confuse them:**
> - **Embedded `zenoh-c`** — the Zenoh C library that the OM1 Go runtime links against for its own messaging (`internal/zenoh`). It is downloaded automatically by `make deps` (the `download-zenohc` target); you don't install it separately.
> - **`zenoh-bridge-ros2dds`** — a standalone bridge that connects a ROS 2 / DDS network to Zenoh. You install this only when integrating OM1 with an existing ROS 2 system, per the [Zenoh Bridge](./zenoh-bridge.md) guide below.

## Middleware components
The following guides walk you through installing and configuring the supported middleware implementations:
- [CycloneDDS](./cyclonedds.md): Install and configure the CycloneDDS RMW implementation for ROS 2.
- [ROS 2 Humble](./ros2-humble.md): Set up the ROS 2 Humble distribution, including core tools and environment configuration.
- [Zenoh Bridge](./zenoh-bridge.md): Install and configure the Zenoh bridge for integrating ROS 2 with Zenoh-based systems.
