---
title: Backgrounds
description: "Backgrounds"
icon: gear
---

### Background Tasks

The Background Tasks system provides a framework for running continuous, long-running processes that operate independently of the main control loop. These tasks typically handle sensor data collection, state monitoring, and other background operations.

Key components:

- **BackgroundOrchestrator**: Manages the lifecycle of all background tasks, including startup and graceful shutdown.
- **Plugins**: Background tasks are loaded dynamically from the `backgrounds/plugins` directory.
- Each background task runs in its own thread, with thread pooling for efficient resource utilization.

Available background tasks include:

- **GPS**: Handles GPS data processing
- **ODOM**: Manages odometry data
- **RF Mapper**: Implements RF signal mapping functionality
- **RPLIDAR**: Interfaces with RPLIDAR sensors
- **RTK**: Real-Time Kinematic positioning
- **Unitree Go2 State**: Manages state for Unitree Go2 robots

Background tasks are configured through the main runtime configuration and can be extended by adding new plugin modules.

### Scopes: `agent_backgrounds` vs `global_backgrounds`

Background tasks come in two scopes:

- **`agent_backgrounds`** (mode-scoped): declared inside a mode. They start when the
  mode is entered and stop when the mode is exited, so they only run while that
  mode is active. In a single-mode config, top-level `agent_backgrounds` seed the
  one synthesized mode.
- **`global_backgrounds`** (system-wide): declared at the top level of the config.
  They start once when the runtime starts and keep running across every mode
  switch until shutdown. Use this scope for tasks that must observe or act
  continuously regardless of the current mode (mirrors `global_lifecycle_hooks`).

```json5
{
  // ...
  global_backgrounds: [
    { type: "ApproachingPerson" },   // runs in every mode
  ],
  modes: {
    welcome: {
      // ...
      agent_backgrounds: [
        { type: "UnitreeGo2FrontierExploration" },  // runs only in this mode
      ],
    },
  },
}
```

Both scopes use the same `Background` plugin interface and registry — the only
difference is lifecycle. Mode-scoped tasks are torn down and rebuilt on each mode
transition; global tasks are not.
