package llm

import (
	"context"
)

type Message struct {
	Role    string // "system" | "user" | "assistant"
	Content string
}

type ToolCall struct {
	Name      string
	Arguments map[string]any
}

type Response struct {
	TextContent string
	ToolCalls   []ToolCall
	Usage       Usage
}

type Usage struct {
	PromptTokens     int
	CompletionTokens int
}

type LLM interface {
	// Call sends a prompt and conversation history to the model.
	Call(ctx context.Context, prompt string, history []Message) (*Response, error)

	// SetSchemas configures the tool schemas that the LLM can use for tool calls.
	SetSchemas(schemas []map[string]any)

	// FunctionSchemas returns the currently configured tool schemas.
	FunctionSchemas() []map[string]any
}

// Factory is a function type for creating LLM instances with a given configuration.
type Factory func(cfg map[string]any) (LLM, error)

// registry holds the mapping of LLM type names to their corresponding factory functions.
var registry = map[string]Factory{}

// Register adds a new LLM factory to the registry under the specified type name.
func Register(typeName string, f Factory) {
	registry[typeName] = f
}

// Load creates an LLM instance based on the provided type name and configuration.
func Load(typeName string, cfg map[string]any) (LLM, error) {
	f, ok := registry[typeName]
	if !ok {
		return nil, &UnknownPluginError{Name: typeName}
	}
	return f(cfg)
}

// UnknownPluginError is returned when an attempt is made to load an LLM plugin that is not registered.
type UnknownPluginError struct{ Name string }

// Error returns a descriptive error message indicating that the specified LLM plugin was not found.
func (e *UnknownPluginError) Error() string { return "llm plugin not found: " + e.Name }
