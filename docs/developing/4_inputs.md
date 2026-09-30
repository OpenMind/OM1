---
title: Inputs
description: "Input Plugin Overview"
icon: pen
---

"Input Plugins" provide the sensory capabilities that allow robots to perceive their environment. These plugins capture, process, and format various types of input data, making them available to the robot's core runtime for decision-making.

## Basic Architecture

- `Sensor` interface defines the core contract for all input plugins ([internal/inputs/sensor.go](https://github.com/OpenMind/OM1/blob/main/internal/inputs/sensor.go))
- `Orchestrator` (`internal/inputs/orchestrator.go`) manages multiple input sources
- Custom input plugins implement the `Sensor` interface and register themselves with `inputs.Register(...)`

```go
// Sensor is the base interface for all input sensors.
type Sensor interface {
    // Listen creates a channel that continuously yields raw input events.
    Listen(ctx context.Context) (<-chan any, error)

    // Poll retrieves a single raw input event.
    Poll(ctx context.Context) (any, error)

    // RawToText converts raw input data into Message format.
    RawToText(ctx context.Context, rawInput any) (*Message, error)

    // FormattedLatestBuffer returns the formatted buffer string.
    FormattedLatestBuffer() string

    // Stop signals the sensor to stop listening and clean up resources.
    Stop()
}
```

## Available Plugins

The authoritative list of plugins is in the [`plugins/inputs`](https://github.com/OpenMind/OM1/blob/main/plugins/inputs) directory. Key plugins include:

- **`GoogleASR`**: Streaming voice-to-text via Google Cloud Speech-to-Text.
- **`FacePresence`**: Detects when a known person's face is visible.
- **Vision Language Model (VLM)** inputs (see below).

Learn how to build a new input plugin [here](../developer_cookbook/input.md).

### Vision Language Model (VLM) Inputs

VLM inputs are sensors that use a vision language model to describe the scene from a camera or RTSP stream. Unlike the VLM backgrounds, which publish descriptions continuously, the VLM inputs are polled by the agent's control loop when it needs to see.

Several VLM providers are available:

| Type              | Provider / Model                                 |
| ----------------- | ------------------------------------------------ |
| `VLMOpenAI`       | OpenAI (`gpt-4-vision-preview`)                  |
| `VLMGemini`       | Google Gemini (`gemini-pro-vision`)              |
| `VLMCosmos`       | Local NVIDIA Cosmos (`nvidia/Cosmos3-Edge`)      |

Each provider has a camera variant (e.g. `VLMOpenAI`) and an RTSP variant (e.g. `VLMOpenAIRTSP`). See the [VLM Backgrounds documentation](./8_backgrounds.md#running-vlmosmos-locally) for instructions on running the `VLMCosmos` model locally.

#### Configuration

All VLM inputs share this common configuration structure:

```json5
{
  "type": "VLMCosmos", // or VLMOpenAI, VLMGemini, etc.
  "api_key": "...",   // required
  "base_url": "...",  // optional; overrides the default endpoint
  "model": "...",     // optional; overrides the default model
  "prompt": "...",    // optional; overrides the default system prompt
  "max_tokens": 128,  // optional
  "resolution_width": 640,
  "resolution_height": 480,
  
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
- `resolution_width`, `resolution_height` (integers): The resolution for frame capture.
- `timeout_sec` (float): The timeout in seconds for requests to the VLM.
- `warmup` (boolean): If true, the plugin will repeatedly try to contact the VLM endpoint in the background after starting up. This is useful for models that take a long time to load. Defaults to `true` for `VLMCosmos` and `false` for others.
- `extra_body` (object): A JSON object of extra parameters to include in the body of the request to the VLM API. This can be used to pass non-standard or provider-specific parameters like `temperature`.
