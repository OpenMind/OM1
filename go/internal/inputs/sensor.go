package inputs

import (
	"context"
)

type Message struct {
	Text    string
	RawText any
}

type Sensor interface {
	RawToText(ctx context.Context, rawText any) (*Message, error)

	Listen(ctx context.Context) (<-chan any, error)

	LatestBuffer() string

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
