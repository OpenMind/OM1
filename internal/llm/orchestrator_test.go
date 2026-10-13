package llm

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewOrchestratorParsesHistoryLength(t *testing.T) {
	f := &fakeLLM{}
	schemas := []map[string]any{{"type": "function"}}
	o := NewOrchestrator(f, map[string]any{"history_length": float64(4)}, schemas)
	require.Equal(t, 4, o.maxLen)
	require.Equal(t, schemas, f.schemas, "schemas are pushed to the underlying LLM")
}

func TestNewOrchestratorDefaultsAndNilConfig(t *testing.T) {
	require.Equal(t, 0, NewOrchestrator(&fakeLLM{}, nil, nil).maxLen)
	require.Equal(t, 0, NewOrchestrator(&fakeLLM{}, map[string]any{}, nil).maxLen, "missing history_length → 0")
}

func TestOrchestratorCallStateless(t *testing.T) {
	f := &fakeLLM{resp: &Response{TextContent: "hi"}}
	o := NewOrchestrator(f, nil, nil)

	resp, err := o.Call(context.Background(), "hello", nil)
	require.NoError(t, err)
	require.Equal(t, "hi", resp.TextContent)
	require.Len(t, f.calls, 1)
	require.Nil(t, f.calls[0].history, "with maxLen 0 no history is passed")
	require.Empty(t, o.msgs, "stateless mode keeps no history")
}

func TestOrchestratorCallAccumulatesHistory(t *testing.T) {
	f := &fakeLLM{resp: &Response{TextContent: "first answer"}}
	o := NewOrchestrator(f, map[string]any{"history_length": float64(5)}, nil)

	_, err := o.Call(context.Background(), "q1", nil)
	require.NoError(t, err)
	require.Len(t, o.msgs, 2)
	require.Equal(t, Message{Role: "user", Content: "q1"}, o.msgs[0])
	require.Equal(t, Message{Role: "assistant", Content: "first answer"}, o.msgs[1])

	f.resp = &Response{TextContent: "second answer"}
	_, err = o.Call(context.Background(), "q2", nil)
	require.NoError(t, err)
	require.Len(t, f.calls[1].history, 2, "second call sees the first turn")
	require.Len(t, o.msgs, 4)
}

func TestOrchestratorRecordsToolCallsWhenNoText(t *testing.T) {
	f := &fakeLLM{resp: &Response{ToolCalls: []ToolCall{{Name: "speak", Arguments: map[string]any{"text": "hi"}}}}}
	o := NewOrchestrator(f, map[string]any{"history_length": float64(3)}, nil)

	_, err := o.Call(context.Background(), "say hi", nil)
	require.NoError(t, err)
	require.Len(t, o.msgs, 2)
	require.Equal(t, "assistant", o.msgs[1].Role)
	require.Equal(t, `speak({"text":"hi"})`, o.msgs[1].Content)
}

func TestOrchestratorHistoryTrimming(t *testing.T) {
	f := &fakeLLM{resp: &Response{TextContent: "ok"}}
	o := NewOrchestrator(f, map[string]any{"history_length": float64(2)}, nil)

	for i := 0; i < 5; i++ {
		_, err := o.Call(context.Background(), "q", nil)
		require.NoError(t, err)
	}
	require.Len(t, o.msgs, 4, "history is capped at maxLen*2 messages")
}

func TestOrchestratorCallPropagatesError(t *testing.T) {
	f := &fakeLLM{err: errors.New("boom")}
	o := NewOrchestrator(f, map[string]any{"history_length": float64(2)}, nil)

	_, err := o.Call(context.Background(), "q", nil)
	require.Error(t, err)
	require.Empty(t, o.msgs, "no history recorded on error")
}

func TestOrchestratorReset(t *testing.T) {
	f := &fakeLLM{resp: &Response{TextContent: "ok"}}
	o := NewOrchestrator(f, map[string]any{"history_length": float64(3)}, nil)
	_, _ = o.Call(context.Background(), "q", nil)
	require.NotEmpty(t, o.msgs)

	o.Reset()
	require.Empty(t, o.msgs)
}

func TestOrchestratorSchemaDelegation(t *testing.T) {
	f := &fakeLLM{}
	o := NewOrchestrator(f, nil, nil)
	schemas := []map[string]any{{"type": "function", "name": "x"}}
	o.SetSchemas(schemas)
	require.Equal(t, schemas, f.schemas)
	require.Equal(t, schemas, o.FunctionSchemas())
}

func TestFormatToolCalls(t *testing.T) {
	out := formatToolCalls([]ToolCall{
		{Name: "speak", Arguments: map[string]any{"text": "hi"}},
		{Name: "move", Arguments: map[string]any{"dir": "left"}},
	})
	require.Equal(t, `speak({"text":"hi"}) | move({"dir":"left"})`, out)
}

func TestOrchestratorCallPartsSendsSystemOnceAndStoresSummary(t *testing.T) {
	f := &fakeLLM{resp: &Response{ToolCalls: []ToolCall{{Name: "wave", Arguments: map[string]any{}}}}}
	o := NewOrchestrator(f, map[string]any{"history_length": float64(3)}, nil)

	_, err := o.CallParts(context.Background(), "SYSTEM", "USER 1 full prompt", "At 10:00:00 you observed: change")
	require.NoError(t, err)
	require.Len(t, f.calls, 1)
	require.Equal(t, "USER 1 full prompt", f.calls[0].prompt)
	require.Equal(t, []Message{{Role: "system", Content: "SYSTEM"}}, f.calls[0].history, "first call: system message only")

	require.Len(t, o.msgs, 2)
	require.Equal(t, Message{Role: "user", Content: "At 10:00:00 you observed: change"}, o.msgs[0], "history keeps the summary, not the prompt")
	require.Equal(t, "assistant", o.msgs[1].Role)
	require.Contains(t, o.msgs[1].Content, "you did: wave")

	f.resp = &Response{}
	_, err = o.CallParts(context.Background(), "SYSTEM", "USER 2", "At 10:00:02 there were no new observations.")
	require.NoError(t, err)
	h := f.calls[1].history
	require.Equal(t, "system", h[0].Role)
	require.Equal(t, 3, len(h), "system + one remembered turn")
	for _, m := range h[1:] {
		require.NotEqual(t, "system", m.Role, "the system message is sent once, never stored in history")
		require.NotContains(t, m.Content, "full prompt")
	}
	require.Equal(t, "you did nothing", o.msgs[3].Content)
}

func TestOrchestratorCallPartsStateless(t *testing.T) {
	f := &fakeLLM{resp: &Response{TextContent: "ok"}}
	o := NewOrchestrator(f, nil, nil)
	_, err := o.CallParts(context.Background(), "SYSTEM", "USER", "summary")
	require.NoError(t, err)
	require.Equal(t, []Message{{Role: "system", Content: "SYSTEM"}}, f.calls[0].history)
	require.Empty(t, o.msgs)
}
