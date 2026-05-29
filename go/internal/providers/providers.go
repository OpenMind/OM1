package providers

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/openmind/om1/internal/llm"
)

type IOProvider struct {
	mu            sync.Mutex
	lastTickStart time.Time
	totalTicks    int64
}

var ioOnce sync.Once
var ioInstance *IOProvider

func IO() *IOProvider {
	ioOnce.Do(func() { ioInstance = &IOProvider{} })
	return ioInstance
}

func (p *IOProvider) RecordTick(start time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastTickStart = start
	p.totalTicks++
}

func (p *IOProvider) TotalTicks() int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.totalTicks
}

// HistoryManager wraps an llm.LLM and manages conversation history automatically,
// mirroring Python's LLMHistoryManager.update_history() decorator on ask().
// When maxLen is 0, history is disabled and calls are delegated directly.
type HistoryManager struct {
	inner  llm.LLM
	mu     sync.Mutex
	msgs   []llm.Message
	maxLen int
}

func NewHistoryManager(inner llm.LLM, config map[string]any) *HistoryManager {
	historyLen := 0
	if config != nil {
		if v, ok := config["history_length"]; ok {
			if n, ok := v.(float64); ok {
				historyLen = int(n)
			}
		}
	}
	return &HistoryManager{inner: inner, maxLen: historyLen}
}

func (h *HistoryManager) SetSchemas(schemas []map[string]any) { h.inner.SetSchemas(schemas) }
func (h *HistoryManager) FunctionSchemas() []map[string]any   { return h.inner.FunctionSchemas() }

// Call injects the accumulated history into the inner LLM call, then records
// the new user prompt and assistant response. On error the turn is not recorded,
// mirroring Python's behaviour of popping an unpaired user message on failure.
func (h *HistoryManager) Call(ctx context.Context, prompt string, _ []llm.Message) (*llm.Response, error) {
	if h.maxLen == 0 {
		return h.inner.Call(ctx, prompt, nil)
	}

	h.mu.Lock()
	snapshot := make([]llm.Message, len(h.msgs))
	copy(snapshot, h.msgs)
	h.mu.Unlock()

	resp, err := h.inner.Call(ctx, prompt, snapshot)
	if err != nil {
		return nil, err
	}

	h.mu.Lock()
	h.msgs = append(h.msgs, llm.Message{Role: "user", Content: prompt})
	if resp.TextContent != "" {
		h.msgs = append(h.msgs, llm.Message{Role: "assistant", Content: resp.TextContent})
	} else if len(resp.ToolCalls) > 0 {
		h.msgs = append(h.msgs, llm.Message{Role: "assistant", Content: formatToolCalls(resp.ToolCalls)})
	}
	if len(h.msgs) > h.maxLen*2 {
		h.msgs = h.msgs[len(h.msgs)-h.maxLen*2:]
	}
	h.mu.Unlock()

	return resp, nil
}

func (h *HistoryManager) Reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.msgs = nil
}

func formatToolCalls(calls []llm.ToolCall) string {
	parts := make([]string, 0, len(calls))
	for _, tc := range calls {
		args, _ := json.Marshal(tc.Arguments)
		parts = append(parts, tc.Name+"("+string(args)+")")
	}
	return strings.Join(parts, " | ")
}
