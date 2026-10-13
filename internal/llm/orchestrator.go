package llm

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
)

// Orchestrator manages conversation history and delegates calls to an LLM.
type Orchestrator struct {
	llm    LLM
	mu     sync.Mutex
	msgs   []Message
	maxLen int
}

// NewOrchestrator creates an Orchestrator with the given LLM, config, and schemas.
func NewOrchestrator(llm LLM, config map[string]any, schemas []map[string]any) *Orchestrator {
	historyLen := 0
	if config != nil {
		if v, ok := config["history_length"]; ok {
			if n, ok := v.(float64); ok {
				historyLen = int(n)
			}
		}
	}

	orchestrator := &Orchestrator{llm: llm, maxLen: historyLen}

	if len(schemas) > 0 {
		orchestrator.llm.SetSchemas(schemas)
	}

	return orchestrator
}

// SetSchemas updates the tool schemas in the underlying LLM.
func (o *Orchestrator) SetSchemas(schemas []map[string]any) { o.llm.SetSchemas(schemas) }

// FunctionSchemas returns the currently configured tool schemas from the underlying LLM.
func (o *Orchestrator) FunctionSchemas() []map[string]any { return o.llm.FunctionSchemas() }

// Call implements the LLM interface. It manages conversation history based on maxLen.
func (o *Orchestrator) Call(ctx context.Context, prompt string, _ []Message) (*Response, error) {
	if o.maxLen == 0 {
		return o.llm.Call(ctx, prompt, nil)
	}

	o.mu.Lock()
	snapshot := make([]Message, len(o.msgs))
	copy(snapshot, o.msgs)
	o.mu.Unlock()

	resp, err := o.llm.Call(ctx, prompt, snapshot)
	if err != nil {
		return nil, err
	}

	o.mu.Lock()
	o.msgs = append(o.msgs, Message{Role: "user", Content: prompt})

	if resp.TextContent != "" {
		o.msgs = append(o.msgs, Message{Role: "assistant", Content: resp.TextContent})
	} else if len(resp.ToolCalls) > 0 {
		o.msgs = append(o.msgs, Message{Role: "assistant", Content: formatToolCalls(resp.ToolCalls)})
	}

	if len(o.msgs) > o.maxLen*2 {
		o.msgs = o.msgs[len(o.msgs)-o.maxLen*2:]
	}

	o.mu.Unlock()

	return resp, nil
}

// CallParts sends system as the system message, the stored history, and user as the new user message.
// Only summary (one line describing the turn's observations) and the model's response are kept in history,
// so the constant part of the prompt is sent once per call rather than once per remembered turn.
func (o *Orchestrator) CallParts(ctx context.Context, system, user, summary string) (*Response, error) {
	if o.maxLen == 0 {
		return o.llm.Call(ctx, user, []Message{{Role: "system", Content: system}})
	}

	o.mu.Lock()
	snapshot := make([]Message, 0, len(o.msgs)+1)
	snapshot = append(snapshot, Message{Role: "system", Content: system})
	snapshot = append(snapshot, o.msgs...)
	o.mu.Unlock()

	resp, err := o.llm.Call(ctx, user, snapshot)
	if err != nil {
		return nil, err
	}

	if summary == "" {
		summary = user
	}

	o.mu.Lock()
	o.msgs = append(o.msgs, Message{Role: "user", Content: summary})

	switch {
	case resp.TextContent != "" && len(resp.ToolCalls) > 0:
		o.msgs = append(o.msgs, Message{Role: "assistant", Content: resp.TextContent + " | you did: " + formatToolCalls(resp.ToolCalls)})
	case resp.TextContent != "":
		o.msgs = append(o.msgs, Message{Role: "assistant", Content: resp.TextContent})
	case len(resp.ToolCalls) > 0:
		o.msgs = append(o.msgs, Message{Role: "assistant", Content: "you did: " + formatToolCalls(resp.ToolCalls)})
	default:
		o.msgs = append(o.msgs, Message{Role: "assistant", Content: "you did nothing"})
	}

	if len(o.msgs) > o.maxLen*2 {
		o.msgs = o.msgs[len(o.msgs)-o.maxLen*2:]
	}

	o.mu.Unlock()

	return resp, nil
}

// Reset clears the conversation history in the Orchestrator.
func (o *Orchestrator) Reset() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.msgs = nil
}

// formatToolCalls converts a list of ToolCalls into a readable string format for conversation history.
func formatToolCalls(calls []ToolCall) string {
	parts := make([]string, 0, len(calls))

	for _, tc := range calls {
		args, _ := json.Marshal(tc.Arguments)
		parts = append(parts, tc.Name+"("+string(args)+")")
	}

	return strings.Join(parts, " | ")
}
