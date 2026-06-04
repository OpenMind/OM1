---
title: Inputs
description: "Input Plugin Overview"
icon: pen
---

"Input Plugins" provide the sensory capabilities that allow robots to perceive their environment. These plugins capture, process, and format various types of input data, making them available to the robot's core runtime for decision-making.

## Basic Architecture

- `Sensor` interface defines the core contract for all input plugins ([internal/inputs/sensor.go](https://github.com/OpenMind/OM1/blob/main/internal/inputs/sensor.go))
- `InputOrchestrator` manages multiple input sources
- Custom input plugins implement the `Sensor` interface

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

## Examples

[Input plugin code examples](https://github.com/OpenMind/OM1/blob/main/plugins/inputs)

Here are a few examples for you to reuse and build on:

- [Google ASR](https://github.com/openmind/OM1/blob/main/plugins/inputs/google_asr/google_asr.go)
- [Face Presence](https://github.com/openmind/OM1/blob/main/plugins/inputs/face_presence/face_presence.go)
- [VLM COCO Local](https://github.com/openmind/OM1/blob/main/plugins/inputs/vlm_coco_local/vlm_coco_local.go)

Learn how to build a new input plugin [here](../developer_cookbook/input.md)
