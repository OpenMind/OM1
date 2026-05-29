package backgrounds

import "context"

// Background represents a long-running process that performs side effects and can be started and stopped by the runtime.
type Background interface {
	Run(ctx context.Context)
	Stop()
}

// Factory creates a Background from a decoded config map.
type Factory func(cfg map[string]any) (Background, error)

// Registry for background factories, keyed by type name.
var registry = map[string]Factory{}

// Register adds a background factory to the registry under the given key.
func Register(typeName string, f Factory) {
	registry[typeName] = f
}

// Load creates a Background for the given type name and config.
func Load(typeName string, cfg map[string]any) (Background, error) {
	f, ok := registry[typeName]
	if !ok {
		return nil, &UnknownPluginError{Name: typeName}
	}
	return f(cfg)
}

// UnknownPluginError is returned when a requested plugin type is not found in the registry.
type UnknownPluginError struct{ Name string }

func (e *UnknownPluginError) Error() string { return "background plugin not found: " + e.Name }
