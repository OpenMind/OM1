package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openmind/om1/internal/llm"
	"github.com/stretchr/testify/require"
)

func captureServer(t *testing.T, status int, responseJSON string) (*httptest.Server, *capturedRequest) {
	t.Helper()
	cap := &capturedRequest{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.path = r.URL.Path
		cap.auth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &cap.body)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(responseJSON))
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

type capturedRequest struct {
	path string
	auth string
	body map[string]any
}

func TestProviderDefaults(t *testing.T) {
	cases := []struct {
		name    string
		cfg     map[string]any
		model   string
		baseURL string
	}{
		{"GeminiLLM", map[string]any{"api_key": "k"}, defaultGeminiModel, defaultGeminiBaseURL},
		{"OpenAILLM", map[string]any{"api_key": "k"}, defaultOpenAIModel, defaultOpenAIBaseURL},
		{"DeepSeekLLM", map[string]any{"api_key": "k"}, defaultDeepSeekModel, defaultDeepSeekBaseURL},
		{"XAILLM", map[string]any{"api_key": "k"}, defaultXAIModel, defaultXAIBaseURL},
		{"OpenRouter", map[string]any{"api_key": "k"}, defaultOpenRouterModel, defaultOpenRouterBaseURL},
		{"NearAILLM", map[string]any{"api_key": "k"}, defaultNearAIModel, defaultNearAIBaseURL},
		{"FunctionGemmaLLM", map[string]any{"api_key": "k"}, defaultFunctionGemmaModel, defaultFunctionGemmaBaseURL},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			instance, err := llm.Load(tc.name, tc.cfg)
			require.NoError(t, err)
			c, ok := instance.(*openAICompatLLM)
			require.Truef(t, ok, "expected *openAICompatLLM, got %T", instance)
			require.Equal(t, tc.model, c.config.Model)
			require.Equal(t, tc.baseURL, c.config.BaseURL)
		})
	}
}

func TestConfigOverridesDefaults(t *testing.T) {
	instance, err := llm.Load("OpenAILLM", map[string]any{
		"api_key":  "k",
		"model":    "gpt-5.2",
		"base_url": "https://example.com/v1",
	})
	require.NoError(t, err)
	c := instance.(*openAICompatLLM)
	require.Equal(t, "gpt-5.2", c.config.Model)
	require.Equal(t, "https://example.com/v1", c.config.BaseURL)
}

func newTestCompat(t *testing.T, baseURL, toolChoice string) *openAICompatLLM {
	t.Helper()
	c, err := newOpenAICompat("TestLLM", map[string]any{
		"api_key":  "test-key",
		"model":    "test-model",
		"base_url": baseURL,
	}, "default-model", "https://unused", toolChoice, true)
	require.NoError(t, err)
	return c
}

func TestOpenAICompatCallParsesResponse(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{
		"choices":[{"message":{"content":"hi there","tool_calls":[
			{"function":{"name":"speak","arguments":"{\"text\":\"woof\"}"}}
		]}}],
		"usage":{"prompt_tokens":12,"completion_tokens":7}
	}`)

	c := newTestCompat(t, srv.URL, "auto")
	c.SetSchemas([]map[string]any{{"type": "function", "function": map[string]any{"name": "speak"}}})

	resp, err := c.Call(context.Background(), "hello", []llm.Message{{Role: "assistant", Content: "prev"}})
	require.NoError(t, err)
	require.NotNil(t, resp)

	require.Equal(t, "/chat/completions", cap.path)
	require.Equal(t, "Bearer test-key", cap.auth)
	require.Equal(t, "test-model", cap.body["model"])
	require.Equal(t, "auto", cap.body["tool_choice"])
	require.Contains(t, cap.body, "tools")

	messages := cap.body["messages"].([]any)
	require.Len(t, messages, 2)
	require.Equal(t, "assistant", messages[0].(map[string]any)["role"])
	last := messages[1].(map[string]any)
	require.Equal(t, "user", last["role"])
	require.Equal(t, "hello", last["content"])

	require.Equal(t, "hi there", resp.TextContent)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "speak", resp.ToolCalls[0].Name)
	require.Equal(t, "woof", resp.ToolCalls[0].Arguments["text"])
	require.Equal(t, 12, resp.Usage.PromptTokens)
	require.Equal(t, 7, resp.Usage.CompletionTokens)
}

func TestOpenAICompatCallOmitsToolsWithoutSchemas(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{"choices":[{"message":{"content":"ok"}}]}`)

	c := newTestCompat(t, srv.URL, "auto")
	resp, err := c.Call(context.Background(), "hello", nil)
	require.NoError(t, err)
	require.Equal(t, "ok", resp.TextContent)
	require.NotContains(t, cap.body, "tools")
	require.NotContains(t, cap.body, "tool_choice")
}

func TestOpenAICompatCallMergesExtraBody(t *testing.T) {
	srv, cap := captureServer(t, http.StatusOK, `{"choices":[{"message":{"content":"ok"}}]}`)

	c := newTestCompat(t, srv.URL, "auto")
	c.extraBody = map[string]any{"temperature": 0.5, "max_tokens": 10}

	_, err := c.Call(context.Background(), "hello", nil)
	require.NoError(t, err)
	require.Equal(t, 0.5, cap.body["temperature"])
	require.Equal(t, float64(10), cap.body["max_tokens"])
}

func TestOpenAICompatCallErrorStatus(t *testing.T) {
	srv, _ := captureServer(t, http.StatusInternalServerError, `{"error":"boom"}`)

	c := newTestCompat(t, srv.URL, "auto")
	_, err := c.Call(context.Background(), "hello", nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "500")
}

func TestOpenAICompatCallEmptyChoices(t *testing.T) {
	srv, _ := captureServer(t, http.StatusOK, `{"choices":[]}`)

	c := newTestCompat(t, srv.URL, "auto")
	_, err := c.Call(context.Background(), "hello", nil)
	require.Error(t, err)
}
