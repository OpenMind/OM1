package llm

import (
	"context"
	"net/http"
	"testing"

	"github.com/openmind/om1/internal/llm"
	"github.com/stretchr/testify/require"
)

func TestQwenDefaultsAndReasoning(t *testing.T) {
	instance, err := llm.Load("QwenLLM", map[string]any{})
	require.NoError(t, err)
	q := instance.(*qwenLLM)
	require.Equal(t, defaultQwenAPIKey, q.config.APIKey)
	require.Equal(t, defaultQwenModel, q.config.Model)
	require.Equal(t, "required", q.toolChoice)
	require.False(t, q.enableReasoning)

	withReasoning, err := llm.Load("QwenLLM", map[string]any{"enable_reasoning": true})
	require.NoError(t, err)
	require.True(t, withReasoning.(*qwenLLM).enableReasoning)
}

func TestParseQwenToolCalls(t *testing.T) {
	text := `thinking... <tool_call>{"name": "speak", "arguments": {"text": "hello"}}</tool_call> ` +
		`and <tool_call>{"name": "move", "arguments": {"dir": "forward"}}</tool_call>`
	calls := parseQwenToolCalls(text)
	require.Len(t, calls, 2)
	require.Equal(t, "speak", calls[0].Name)
	require.Equal(t, "hello", calls[0].Arguments["text"])
	require.Equal(t, "move", calls[1].Name)
	require.Equal(t, "forward", calls[1].Arguments["dir"])

	require.Empty(t, parseQwenToolCalls(`<tool_call>{bad json}</tool_call><tool_call>{"arguments":{}}</tool_call>`))
}

func newTestQwen(t *testing.T, baseURL string, reasoning bool) *qwenLLM {
	t.Helper()
	instance, err := NewQwen(map[string]any{
		"base_url":         baseURL,
		"enable_reasoning": reasoning,
	})
	require.NoError(t, err)
	return instance.(*qwenLLM)
}

func lastMessageContent(t *testing.T, cap *capturedRequest) string {
	t.Helper()
	messages := cap.body["messages"].([]any)
	require.NotEmpty(t, messages)
	return messages[len(messages)-1].(map[string]any)["content"].(string)
}

func TestQwenCallAppendsNoThink(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{"choices":[{"message":{"content":"ok"}}]}`)

	q := newTestQwen(t, srv.URL, false)
	q.SetSchemas([]map[string]any{{"type": "function", "function": map[string]any{"name": "speak"}}})

	_, err := q.Call(context.Background(), "hello", nil)
	require.NoError(t, err)

	require.Equal(t, "hello /no_think", lastMessageContent(t, cap))
	require.Equal(t, "required", cap.body["tool_choice"])
	require.Contains(t, cap.body, "chat_template_kwargs")
}

func TestQwenCallReasoningKeepsPromptVerbatim(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{"choices":[{"message":{"content":"ok"}}]}`)

	q := newTestQwen(t, srv.URL, true)
	_, err := q.Call(context.Background(), "hello", nil)
	require.NoError(t, err)
	require.Equal(t, "hello", lastMessageContent(t, cap))
}

func TestQwenCallParsesTextToolCalls(t *testing.T) {
	srv, _ := captureServer(t, http.StatusOK, `{"choices":[{"message":{"content":
		"sure <tool_call>{\"name\":\"move\",\"arguments\":{\"dir\":\"left\"}}</tool_call>"}}]}`)

	q := newTestQwen(t, srv.URL, false)
	resp, err := q.Call(context.Background(), "go left", nil)
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "move", resp.ToolCalls[0].Name)
	require.Equal(t, "left", resp.ToolCalls[0].Arguments["dir"])
}

func TestQwenCallPrefersStructuredToolCalls(t *testing.T) {
	// Structured tool_calls present: the text fallback must not run.
	srv, _ := captureServer(t, http.StatusOK, `{"choices":[{"message":{
		"content":"<tool_call>{\"name\":\"ignored\",\"arguments\":{}}</tool_call>",
		"tool_calls":[{"function":{"name":"speak","arguments":"{\"text\":\"hi\"}"}}]
	}}]}`)

	q := newTestQwen(t, srv.URL, false)
	resp, err := q.Call(context.Background(), "say hi", nil)
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "speak", resp.ToolCalls[0].Name)
	require.Equal(t, "hi", resp.ToolCalls[0].Arguments["text"])
}
