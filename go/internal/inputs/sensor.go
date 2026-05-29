package inputs

import (
	"context"
	"time"
)

type Message struct {
	Timestamp float64
	Message   string
}

func NewMessage(text string) *Message {
	return &Message{
		Timestamp: float64(time.Now().UnixNano()) / 1e9,
		Message:   text,
	}
}

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

type Factory func(cfg map[string]any) (Sensor, error)

var registry = map[string]Factory{}

func Register(typeName string, f Factory) {
	registry[typeName] = f
}

func Load(typeName string, cfg map[string]any) (Sensor, error) {
	f, ok := registry[typeName]
	if !ok {
		return nil, &UnknownPluginError{Kind: "input", Name: typeName}
	}
	return f(cfg)
}

type UnknownPluginError struct {
	Kind string
	Name string
}

func (e *UnknownPluginError) Error() string {
	return e.Kind + " plugin not found: " + e.Name
}
