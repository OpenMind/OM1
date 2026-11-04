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
- **`UnitreeGo2FrontierExploration`**: Autonomous frontier exploration for the Unitree Go2.
- **Vision Language Model (VLM)** backgrounds (see below).

> The authoritative list is whatever is registered via `bg.Register(...)` under `plugins/backgrounds/`. Background tasks are configured through the runtime config and can be extended by adding new plugin modules.

### Vision Language Model (VLM) Backgrounds

VLM backgrounds periodically capture frames from a camera or RTSP stream, send them to a vision language model, and publish the model's text description of the scene. This allows an agent to have a general sense of its surroundings.

Several VLM providers are available:

| Type              | Provider / Model                                 |
| ----------------- | ------------------------------------------------ |
| `VLMOpenAI`       | OpenAI (`gpt-4-vision-preview`)                  |
| `VLMGemini`       | Google Gemini (`gemini-pro-vision`)              |
| `VLMCosmos`       | Local NVIDIA Cosmos (`nvidia/Cosmos3-Edge`)      |

Each provider has a camera variant (e.g. `VLMOpenAI`) and an RTSP variant (e.g. `VLMOpenAIRTSP`).

#### Configuration

All VLM backgrounds share this common configuration structure:

```json5
{
  "type": "VLMCosmos", // or VLMOpenAI, VLMGemini, etc.
  "api_key": "...",   // required
  "base_url": "...",  // optional; overrides the default endpoint
  "model": "...",     // optional; overrides the default model
  "prompt": "...",    // optional; overrides the default system prompt
  "max_tokens": 128,  // optional
  "fps": 4,           // optional; frames to process per second
  "resolution_width": 640,
  "resolution_height": 480,
  
  // New in 1.9.0
  "timeout_sec": 10.0,
  "warmup": true,
  "extra_body": { "temperature": 0.5 }
}
```

- `api_key` (string, required): Your API key for the VLM provider.
- `base_url` (string): The base URL of the API endpoint. Defaults to the standard endpoint for each provider.
- `model` (string): The specific model to use (e.g., `gpt-4o`).
- `prompt` (string): The system prompt sent to the model with each frame.
- `max_tokens` (integer): The maximum number of tokens to generate in the response.
- `fps` (integer): The number of frames per second to capture and send to the model.
- `resolution_width`, `resolution_height` (integers): The resolution for frame capture.
- `timeout_sec` (float): The timeout in seconds for requests to the VLM.
- `warmup` (boolean): If true, the plugin will repeatedly try to contact the VLM endpoint in the background after starting up, and will not send any real requests until it succeeds. This is useful for models that take a long time to load. Defaults to `true` for `VLMCosmos` and `false` for others.
- `extra_body` (object): A JSON object of extra parameters to include in the body of the request to the VLM API. This can be used to pass non-standard or provider-specific parameters like `temperature`.

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
