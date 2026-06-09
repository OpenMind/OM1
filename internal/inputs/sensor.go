package inputs

import (
	"context"
	"time"
)

// Message represents a processed input event with a timestamp and text content.
type Message struct {
	Timestamp float64
	Message   string
}

// NewMessage creates a new Message with the current timestamp and the given text.
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

// TickTrigger is an optional interface that sensors can implement to indicate
// whether they should wake the cortex loop when new input is available.
type TickTrigger interface {
	TriggersTick() bool
}

// triggersTick reports whether a sensor has opted in to waking the cortex loop.
func triggersTick(sensor Sensor) bool {
	t, ok := sensor.(TickTrigger)
	return ok && t.TriggersTick()
}

// Factory is a function type for creating Sensor instances with a given configuration.
type Factory func(cfg map[string]any) (Sensor, error)

// registry holds the mapping of Sensor type names to their corresponding factory functions.
var registry = map[string]Factory{}

// Register adds a new Sensor factory to the registry under the specified type name.
func Register(typeName string, f Factory) {
	registry[typeName] = f
}

// Load creates a Sensor instance based on the provided type name and configuration.
func Load(typeName string, cfg map[string]any) (Sensor, error) {
	f, ok := registry[typeName]
	if !ok {
		return nil, &UnknownPluginError{Kind: "input", Name: typeName}
	}
	return f(cfg)
}

// UnknownPluginError is returned when an attempt is made to load a Sensor plugin that is not registered.
type UnknownPluginError struct {
	Kind string
	Name string
}

// Error returns a descriptive error message indicating that the specified plugin type and name were not found.
func (e *UnknownPluginError) Error() string {
	return e.Kind + " plugin not found: " + e.Name
}
