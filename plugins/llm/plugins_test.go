package llm

import (
	"testing"

	"github.com/openmind/om1/internal/llm"
)

// TestCloudProvidersRequireAPIKey ensures the hosted, OpenAI-compatible providers
// reject configs that omit api_key, mirroring the Python plugins.
func TestCloudProvidersRequireAPIKey(t *testing.T) {
	for _, name := range []string{
		"GeminiLLM", "OpenAILLM", "DeepSeekLLM", "XAILLM",
		"OpenRouter", "NearAILLM", "FunctionGemmaLLM",
	} {
		if _, err := llm.Load(name, map[string]any{}); err == nil {
			t.Errorf("%s: expected error when api_key is missing", name)
		}
		if _, err := llm.Load(name, map[string]any{"api_key": "k"}); err != nil {
			t.Errorf("%s: unexpected error with api_key: %v", name, err)
		}
	}
}

// TestLocalProvidersNoAPIKey ensures local providers load without an api_key.
func TestLocalProvidersNoAPIKey(t *testing.T) {
	for _, name := range []string{"QwenLLM", "OllamaLLM"} {
		if _, err := llm.Load(name, map[string]any{}); err != nil {
			t.Errorf("%s: unexpected error: %v", name, err)
		}
	}
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
		instance, err := llm.Load(tc.name, tc.cfg)
		if err != nil {
			t.Fatalf("%s: load: %v", tc.name, err)
		}
		c, ok := instance.(*openAICompatLLM)
		if !ok {
			t.Fatalf("%s: expected *openAICompatLLM, got %T", tc.name, instance)
		}
		if c.config.Model != tc.model {
			t.Errorf("%s: model = %q, want %q", tc.name, c.config.Model, tc.model)
		}
		if c.config.BaseURL != tc.baseURL {
			t.Errorf("%s: base_url = %q, want %q", tc.name, c.config.BaseURL, tc.baseURL)
		}
	}
}

func TestConfigOverridesDefaults(t *testing.T) {
	instance, err := llm.Load("OpenAILLM", map[string]any{
		"api_key":  "k",
		"model":    "gpt-5.2",
		"base_url": "https://example.com/v1",
	})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	c := instance.(*openAICompatLLM)
	if c.config.Model != "gpt-5.2" || c.config.BaseURL != "https://example.com/v1" {
		t.Errorf("overrides not applied: %+v", c.config)
	}
}

func TestQwenDefaultsAndReasoning(t *testing.T) {
	instance, err := llm.Load("QwenLLM", map[string]any{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	q := instance.(*qwenLLM)
	if q.config.APIKey != defaultQwenAPIKey {
		t.Errorf("api_key = %q, want %q", q.config.APIKey, defaultQwenAPIKey)
	}
	if q.config.Model != defaultQwenModel {
		t.Errorf("model = %q, want %q", q.config.Model, defaultQwenModel)
	}
	if q.toolChoice != "required" {
		t.Errorf("tool_choice = %q, want required", q.toolChoice)
	}
	if q.enableReasoning {
		t.Error("enableReasoning should default to false")
	}

	withReasoning, err := llm.Load("QwenLLM", map[string]any{"enable_reasoning": true})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !withReasoning.(*qwenLLM).enableReasoning {
		t.Error("enable_reasoning config not honored")
	}
}

func TestOllamaDefaults(t *testing.T) {
	instance, err := llm.Load("OllamaLLM", map[string]any{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	o := instance.(*ollamaLLM)
	if o.config.Model != defaultOllamaModel {
		t.Errorf("model = %q, want %q", o.config.Model, defaultOllamaModel)
	}
	if o.chatURL != defaultOllamaBaseURL+"/api/chat" {
		t.Errorf("chatURL = %q", o.chatURL)
	}
	if o.config.Temperature != defaultOllamaTemp || o.config.NumCtx != defaultOllamaNumCtx {
		t.Errorf("option defaults not applied: %+v", o.config)
	}
}

func TestParseQwenToolCalls(t *testing.T) {
	text := `thinking... <tool_call>{"name": "speak", "arguments": {"text": "hello"}}</tool_call> ` +
		`and <tool_call>{"name": "move", "arguments": {"dir": "forward"}}</tool_call>`
	calls := parseQwenToolCalls(text)
	if len(calls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(calls))
	}
	if calls[0].Name != "speak" || calls[0].Arguments["text"] != "hello" {
		t.Errorf("first call mismatch: %+v", calls[0])
	}
	if calls[1].Name != "move" || calls[1].Arguments["dir"] != "forward" {
		t.Errorf("second call mismatch: %+v", calls[1])
	}

	// Malformed JSON and nameless blocks are skipped.
	if got := parseQwenToolCalls(`<tool_call>{bad json}</tool_call><tool_call>{"arguments":{}}</tool_call>`); len(got) != 0 {
		t.Errorf("expected 0 calls, got %d", len(got))
	}
}
