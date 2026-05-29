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

// AgentAction bundles the metadata, auto-generated schema, and connector for
// one action type.  Schema is populated automatically by Load from the
// interface registered via RegisterInterface.
type AgentAction struct {
	// Name is the action's package/directory name (e.g. "move").
	Name string

	// LLMLabel is the function name exposed to the LLM (e.g. "move").
	LLMLabel string

	// Schema is the OpenAI-compatible JSON Schema, built automatically from
	// the input struct registered via RegisterInterface.
	Schema map[string]any

	// ExcludeFromPrompt omits this action from the function schema list sent
	// to the LLM.
	ExcludeFromPrompt bool

	Connector Connector
}

// Factory creates a Connector from a decoded config map.
type Factory func(cfg map[string]any) (Connector, error)

var connectorRegistry = map[string]Factory{}

// Register adds a connector factory.  connectorKey is "action/connector"
// (e.g. "move/ros2").  Called from plugin init() functions.
func Register(connectorKey string, factory Factory) {
	connectorRegistry[connectorKey] = factory
}

// Load instantiates the connector for actionName with the given connectorType
// and config map, and returns a fully-formed AgentAction with an auto-generated
// schema.
func Load(actionName, connectorType, llmLabel string, connectorConfig map[string]any) (*AgentAction, error) {
	key := actionName + "/" + connectorType
	factory, ok := connectorRegistry[key]
	if !ok {
		// Fall back to connector-type-only key for generic connectors.
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
