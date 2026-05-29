package llm

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
)

// Orchestrator wraps an LLM and manages conversation history automatically,
// mirroring Python's LLMHistoryManager.update_history() decorator on ask().
// When maxLen is 0, history is disabled and calls are delegated directly.
type Orchestrator struct {
	inner  LLM
	mu     sync.Mutex
	msgs   []Message
	maxLen int
}

// NewOrchestrator creates an Orchestrator that wraps the given LLM with
// history management and schema configuration.
func NewOrchestrator(inner LLM, config map[string]any, schemas []map[string]any) *Orchestrator {
	historyLen := 0
	if config != nil {
		if v, ok := config["history_length"]; ok {
			if n, ok := v.(float64); ok {
				historyLen = int(n)
			}
		}
	}
	o := &Orchestrator{inner: inner, maxLen: historyLen}
	if len(schemas) > 0 {
		o.inner.SetSchemas(schemas)
	}
	return o
}

func (o *Orchestrator) SetSchemas(schemas []map[string]any) { o.inner.SetSchemas(schemas) }
func (o *Orchestrator) FunctionSchemas() []map[string]any   { return o.inner.FunctionSchemas() }

// Call injects the accumulated history into the inner LLM call, then records
// the new user prompt and assistant response. On error the turn is not recorded,
// mirroring Python's behaviour of popping an unpaired user message on failure.
func (o *Orchestrator) Call(ctx context.Context, prompt string, _ []Message) (*Response, error) {
	if o.maxLen == 0 {
		return o.inner.Call(ctx, prompt, nil)
	}

	o.mu.Lock()
	snapshot := make([]Message, len(o.msgs))
	copy(snapshot, o.msgs)
	o.mu.Unlock()

	resp, err := o.inner.Call(ctx, prompt, snapshot)
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

func (o *Orchestrator) Reset() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.msgs = nil
}

func formatToolCalls(calls []ToolCall) string {
	parts := make([]string, 0, len(calls))
	for _, tc := range calls {
		args, _ := json.Marshal(tc.Arguments)
		parts = append(parts, tc.Name+"("+string(args)+")")
	}
	return strings.Join(parts, " | ")
}
