package actions

import "context"

// Input is the typed value parsed from an LLM tool-call argument.
type Input any

// Output is whatever the connector returns after executing an action.
type Output any

// Connector executes a single action decision on hardware or a remote system.
type Connector interface {
	// Connect is called once per LLM decision with the parsed input.
	Connect(ctx context.Context, input Input) (Output, error)

	// Tick is called every runtime cycle for connectors that need a heartbeat.
	Tick(ctx context.Context)

	// Stop signals the connector to release resources.
	Stop()
}

// Factory creates a Connector from a decoded config map.
type AgentAction struct {
	// Name is the unique identifier for this action.
	Name string

	// LLMLabel is the string used in the LLM prompt to refer to this action.
	LLMLabel string

	// Schema is the OpenAI-compatible JSON Schema.
	Schema map[string]any

	// ExcludeFromPrompt, if true, indicates that this action should not be included in the LLM prompt.
	ExcludeFromPrompt bool

	// Connector is the implementation of this action.
	Connector Connector
}

// Factory creates a Connector from a decoded config map.
type Factory func(cfg map[string]any) (Connector, error)

var connectorRegistry = map[string]Factory{}

// Register adds a connector factory to the registry under the given key.
func Register(connectorKey string, factory Factory) {
	connectorRegistry[connectorKey] = factory
}

// Load creates a Connector for the given action and connector types, using the provided config.
func Load(actionName, connectorType, llmLabel string, connectorConfig map[string]any) (*AgentAction, error) {
	key := actionName + "/" + connectorType
	factory, ok := connectorRegistry[key]
	if !ok {
		factory, ok = connectorRegistry[connectorType]
		if !ok {
			return nil, &UnknownPluginError{Kind: "action connector", Name: key}
		}
	}

	connector, err := factory(connectorConfig)
	if err != nil {
		return nil, err
	}

	schema, _ := BuildSchemaForAction(actionName, llmLabel)

	return &AgentAction{
		Name:      actionName,
		LLMLabel:  llmLabel,
		Schema:    schema,
		Connector: connector,
	}, nil
}

// UnknownPluginError is returned when a plugin type name is not registered.
type UnknownPluginError struct {
	Kind string
	Name string
}

func (e *UnknownPluginError) Error() string {
	return e.Kind + " plugin not found: " + e.Name
}
