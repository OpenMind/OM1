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

	// SetSchemas provides the OpenAI-compatible tool schemas that will be
	// attached to every subsequent Call.  The runtime calls this once after
	// loading the action list for the active mode.
	SetSchemas(schemas []map[string]any)

	// FunctionSchemas returns the currently configured tool schemas.
	FunctionSchemas() []map[string]any
}

type Factory func(cfg map[string]any) (LLM, error)

var registry = map[string]Factory{}

func Register(typeName string, f Factory) {
	registry[typeName] = f
}

func Load(typeName string, cfg map[string]any) (LLM, error) {
	f, ok := registry[typeName]
	if !ok {
		return nil, &UnknownPluginError{Name: typeName}
	}
	return f(cfg)
}

type UnknownPluginError struct{ Name string }

func (e *UnknownPluginError) Error() string { return "llm plugin not found: " + e.Name }
