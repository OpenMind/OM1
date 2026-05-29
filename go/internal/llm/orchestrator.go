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

func (o *Orchestrator) SetSchemas(schemas []map[string]any) { o.llm.SetSchemas(schemas) }
func (o *Orchestrator) FunctionSchemas() []map[string]any   { return o.llm.FunctionSchemas() }

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
