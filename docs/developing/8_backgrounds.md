---
title: Backgrounds
description: "Backgrounds"
icon: gear
---

### Background Tasks

The Background Tasks system provides a framework for running continuous, long-running processes that operate independently of the main control loop. These tasks typically handle sensor data collection, state monitoring, and other background operations.

Key components:

- **Orchestrator** (`internal/backgrounds/orchestrator.go`): Manages the lifecycle of all background tasks, including startup and graceful shutdown.
- **Plugins**: Background tasks live under `plugins/backgrounds/` and register themselves via `bg.Register(...)`.
- Each background task runs in its own goroutine (one goroutine per background); the orchestrator waits for all of them to finish on shutdown.

The currently registered background tasks are:

- **`TTSControl`**: Coordinates text-to-speech playback state.
- **`ApproachingPerson`**: Reacts when a person approaches.
- **`VLMGemini`**, **`VLMGeminiRTSP`**: Background vision captioning via Gemini.
- **`VLMOpenAI`**, **`VLMOpenAIRTSP`**: Background vision captioning via OpenAI.
- **`UnitreeGo2FrontierExploration`**: Autonomous frontier exploration for the Unitree Go2.

> The authoritative list is whatever is registered via `bg.Register(...)` under `plugins/backgrounds/`. Background tasks are configured through the runtime config and can be extended by adding new plugin modules.

### Scopes: `agent_backgrounds` vs `global_backgrounds`

Background tasks come in two scopes:

- **`agent_backgrounds`** (mode-scoped): declared inside a mode. They start when the
  mode is entered and stop when the mode is exited, so they only run while that
  mode is active. In a single-mode config, top-level `agent_backgrounds` seed the
  one synthesized mode.
- **`global_backgrounds`** (system-wide): declared at the top level of the config.
  They start once when the runtime starts and keep running across every mode
  switch until shutdown. Use this scope for tasks that must observe or act
  continuously regardless of the current mode.

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

