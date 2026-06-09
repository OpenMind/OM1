---
title: New Input Plugin
description: "Build a new input plugin"
icon: pen
---

## Overview
This guide walks you through creating a new input plugin for OM1. Input plugins allow you to integrate various data sources and sensors into your agent.

## Prerequisites

- Understanding of Go interfaces and structs
- Familiarity with concurrent programming in Go
- Knowledge of the data source you're integrating

## Implementation Steps

### Step 1: Create a Provider (Optional)
If your plugin requires complex initialization or external service integration, create a provider.

Location: `/internal/providers/your_provider.go`

### Step 2: Create a new Plugin File

To proceed with a new input plugin integration, create a Go file.
Location: `/plugins/inputs/your_plugin/your_plugin.go`

Required imports:
```go
package your_plugin

import (
    "context"
    "github.com/openmind/om1/internal/inputs"
)
```

### Step 3: Implement the Sensor Interface

Your plugin must implement the `Sensor` interface defined in `internal/inputs/sensor.go`:

```go
type Sensor interface {
    // Listen creates a channel that continuously yields raw input events.
    Listen(ctx context.Context) (<-chan any, error)

    // Poll retrieves a single raw input event.
    Poll(ctx context.Context) (any, error)

    // RawToText converts raw input data into Message format.
    RawToText(ctx context.Context, rawInput any) (*inputs.Message, error)

    // FormattedLatestBuffer returns the formatted buffer string.
    FormattedLatestBuffer() string

    // Stop signals the sensor to stop listening and clean up resources.
    Stop()
}
```

### Step 4: Implement Your Plugin Struct

```go
type YourInput struct {
    config    map[string]any
    isRunning bool
    buffer    string
}

func New(cfg map[string]any) (inputs.Sensor, error) {
    return &YourInput{
        config: cfg,
    }, nil
}
```

### Step 5: Implement Required Methods

```go
func (y *YourInput) Listen(ctx context.Context) (<-chan any, error) {
    ch := make(chan any)
    go func() {
        defer close(ch)
        for {
            select {
            case <-ctx.Done():
                return
            default:
                // Read from your data source and send to channel
                data, err := y.readData()
                if err == nil {
                    ch <- data
                }
            }
        }
    }()
    return ch, nil
}

func (y *YourInput) Poll(ctx context.Context) (any, error) {
    // Return a single reading from your data source
    return y.readData()
}

func (y *YourInput) RawToText(ctx context.Context, rawInput any) (*inputs.Message, error) {
    // Convert raw input to a Message
    text := formatAsText(rawInput)
    return inputs.NewMessage(text), nil
}

func (y *YourInput) FormattedLatestBuffer() string {
    return y.buffer
}

func (y *YourInput) Stop() {
    y.isRunning = false
}
```

## Plugin Registration

Plugins are registered using the `inputs.Register` function. Add an `init()` function to your plugin:

```go
func init() {
    inputs.Register("YourInput", New)
}
```

### How it works:

- The `inputs.Register` function maps your plugin type name to its factory function
- The `inputs.Load` function creates instances based on configuration
- Plugin type names in config files must match the registered name

### Requirements:

- Implement the `Sensor` interface from `internal/inputs/sensor.go`
- Register your factory function with `inputs.Register`
- File must be in `/plugins/inputs/` directory
