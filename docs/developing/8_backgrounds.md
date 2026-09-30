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
- **`VLMCosmos`**, **`VLMCosmosRTSP`**: Background vision captioning via a local NVIDIA Cosmos3-Edge vLLM server.
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

### Running `VLMCosmos` locally

`VLMCosmos` and `VLMCosmosRTSP` call an NVIDIA Cosmos3-Edge model served by vLLM on `http://127.0.0.1:8000/v1`. The `cosmos_edge` service in `docker-compose.yml` runs it on an NVIDIA GPU host such as Jetson Thor; it sits behind the `cosmos` profile, so a plain `docker compose up` does not start it.

```bash
docker compose --profile cosmos up -d cosmos_edge
VLM_BACKGROUND_PLUGIN=VLMCosmosRTSP OM1_COMMAND=greeting_conversation docker compose up -d om1
```

- The first start downloads the weights (~9 GB) into `~/.cache/huggingface` and compiles the model, which takes several minutes. Later starts reuse the cache in `~/.cache/vllm`.
- OM1 does not wait for the server. The plugin warms the endpoint up in the background and emits no descriptions until it answers, so the first request (~40 s) never reaches the LLM. After that, a description takes about 0.25 s.
- `COSMOS_GPU_MEMORY_UTILIZATION` (default `0.1`) caps the GPU memory share so the server can coexist with other local models.
- Send native-resolution frames for readable text, e.g. `resolution_width: 1280, resolution_height: 720` for a 720p stream; the RTSP default of 480x640 downscales the frame.
