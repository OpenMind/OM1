package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("GeminiLLM", newGemini)
}

type geminiModel string

const (
	GeminiModel25Flash     geminiModel = "gemini-2.5-flash"
	GeminiModel25FlashLite geminiModel = "gemini-2.5-flash-lite"
	GeminiModel25Pro       geminiModel = "gemini-2.5-pro"
	GeminiModel20Flash     geminiModel = "gemini-2.0-flash"
	GeminiModel20FlashLite geminiModel = "gemini-2.0-flash-lite"

	defaultGeminiModel   geminiModel = "gemini-3.1-flash-lite"
	defaultGeminiBaseURL string      = "https://api.openmind.com/api/core/gemini"
)

type geminiConfig struct {
	APIKey     string `json:"api_key"`
	Model      string `json:"model"`
	BaseURL    string `json:"base_url"`
	AgentName  string `json:"agent_name"`
	HistoryLen int    `json:"history_length"`
}

type geminiLLM struct {
	config  geminiConfig
	schemas []map[string]any
}

func newGemini(configMap map[string]any) (llm.LLM, error) {
	var cfg geminiConfig
	if err := remarshal(configMap, &cfg); err != nil {
		return nil, fmt.Errorf("GeminiLLM config: %w", err)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("GeminiLLM: api_key is required")
	}
	if cfg.Model == "" {
		cfg.Model = string(defaultGeminiModel)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultGeminiBaseURL
	}
	return &geminiLLM{config: cfg}, nil
}

func (g *geminiLLM) FunctionSchemas() []map[string]any { return g.schemas }

func (g *geminiLLM) SetSchemas(schemas []map[string]any) { g.schemas = schemas }

func (g *geminiLLM) Call(ctx context.Context, prompt string, history []llm.Message) (*llm.Response, error) {
	messages := buildMessages(prompt, history)

	requestBody := map[string]any{
		"model":    g.config.Model,
		"messages": messages,
	}
	if len(g.schemas) > 0 {
		tools := make([]map[string]any, len(g.schemas))
		for i, schema := range g.schemas {
			tools[i] = map[string]any{"type": "function", "function": schema}
		}
		requestBody["tools"] = tools
		requestBody["tool_choice"] = "auto"
	}

	requestBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("GeminiLLM: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		g.config.BaseURL+"/chat/completions", bytes.NewReader(requestBytes))
	if err != nil {
		return nil, fmt.Errorf("GeminiLLM: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+g.config.APIKey)

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return nil, fmt.Errorf("GeminiLLM: http: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GeminiLLM: api %d: %s", resp.StatusCode, body)
	}

	return parseOpenAIResponse(body)
}
