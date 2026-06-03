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
