package llm

import (
	"context"
	"net/http"
	"testing"

	"github.com/openmind/om1/internal/llm"
	"github.com/stretchr/testify/require"
)

func TestOllamaDefaults(t *testing.T) {
	instance, err := llm.Load("OllamaLLM", map[string]any{})
	require.NoError(t, err)
	o := instance.(*ollamaLLM)
	require.Equal(t, defaultOllamaModel, o.config.Model)
	require.Equal(t, defaultOllamaBaseURL+"/api/chat", o.chatURL)
	require.Equal(t, defaultOllamaTemp, o.config.Temperature)
	require.Equal(t, defaultOllamaNumCtx, o.config.NumCtx)
}

func newTestOllama(t *testing.T, baseURL string) *ollamaLLM {
	t.Helper()
	instance, err := NewOllama(map[string]any{
		"base_url": baseURL,
		"model":    "test-model",
	})
	require.NoError(t, err)
	return instance.(*ollamaLLM)
}

func TestOllamaCallRequestAndParse(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{
		"message":{"content":"woof","tool_calls":[
			{"function":{"name":"emotion","arguments":{"type":"happy"}}}
		]},
		"prompt_eval_count":5,"eval_count":9
	}`)

	o := newTestOllama(t, srv.URL)
	o.SetSchemas([]map[string]any{{"type": "function", "function": map[string]any{"name": "emotion"}}})

	resp, err := o.Call(context.Background(), "hello", nil)
	require.NoError(t, err)

	require.Equal(t, "/api/chat", cap.path)
	require.Equal(t, "test-model", cap.body["model"])
	require.Equal(t, false, cap.body["stream"])
	require.Contains(t, cap.body, "tools")
	options := cap.body["options"].(map[string]any)
	require.Equal(t, defaultOllamaTemp, options["temperature"])
	require.Equal(t, float64(defaultOllamaNumCtx), options["num_ctx"])

	require.Equal(t, "woof", resp.TextContent)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "emotion", resp.ToolCalls[0].Name)
	require.Equal(t, "happy", resp.ToolCalls[0].Arguments["type"])
	require.Equal(t, 5, resp.Usage.PromptTokens)
	require.Equal(t, 9, resp.Usage.CompletionTokens)
}

func TestOllamaCallOmitsToolsWithoutSchemas(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{"message":{"content":"hi"}}`)

	o := newTestOllama(t, srv.URL)
	_, err := o.Call(context.Background(), "hello", nil)
	require.NoError(t, err)
	require.NotContains(t, cap.body, "tools")
}

func TestOllamaCallErrorStatus(t *testing.T) {
	srv, _ := captureServer(t, http.StatusInternalServerError, `nope`)

	o := newTestOllama(t, srv.URL)
	_, err := o.Call(context.Background(), "hello", nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "500")
}

func TestOllamaTrimsTrailingSlash(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{"message":{"content":"hi"}}`)

	instance, err := NewOllama(map[string]any{"base_url": srv.URL + "/", "model": "m"})
	require.NoError(t, err)
	_, err = instance.(*ollamaLLM).Call(context.Background(), "hello", nil)
	require.NoError(t, err)
	require.Equal(t, "/api/chat", cap.path)
}
