package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/metrics"
)

type compatConfig struct {
	APIKey     string `json:"api_key"`
	Model      string `json:"model"`
	BaseURL    string `json:"base_url"`
	AgentName  string `json:"agent_name"`
	HistoryLen int    `json:"history_length"`
}

type openAICompatLLM struct {
	provider   string
	config     compatConfig
	toolChoice string // "auto" or "required"
	extraBody  map[string]any
	schemas    []map[string]any
}

func newOpenAICompat(provider string, configMap map[string]any, defaultModel, defaultBaseURL, toolChoice string, requireAPIKey bool) (*openAICompatLLM, error) {
	var cfg compatConfig
	if err := remarshal(configMap, &cfg); err != nil {
		return nil, fmt.Errorf("%s config: %w", provider, err)
	}
	if requireAPIKey && cfg.APIKey == "" {
		return nil, fmt.Errorf("%s: api_key is required", provider)
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return &openAICompatLLM{provider: provider, config: cfg, toolChoice: toolChoice}, nil
}

func (c *openAICompatLLM) FunctionSchemas() []map[string]any { return c.schemas }

func (c *openAICompatLLM) SetSchemas(schemas []map[string]any) { c.schemas = schemas }

func (c *openAICompatLLM) Call(ctx context.Context, prompt string, history []llm.Message) (*llm.Response, error) {
	// start brackets the whole operation: build + travel + proxy + parse.
	start := time.Now()

	requestBody := map[string]any{
		"model":    c.config.Model,
		"messages": buildMessages(prompt, history),
	}

	if len(c.schemas) > 0 {
		requestBody["tools"] = c.schemas
		requestBody["tool_choice"] = c.toolChoice
	}

	for k, v := range c.extraBody {
		requestBody[k] = v
	}

	body, proxyTotalMs, err := c.doRequest(ctx, requestBody)
	if err != nil {
		return nil, err
	}
	resp, err := parseOpenAIResponse(body)
	metrics.RecordRequestTiming("llm", time.Since(start).Seconds(), proxyTotalMs)
	return resp, err
}

// doRequest sends the request and returns the response body plus the gateway's
// x-proxy-total-ms header value (empty if absent), used for per-request timing.
func (c *openAICompatLLM) doRequest(ctx context.Context, requestBody map[string]any) ([]byte, string, error) {
	requestBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, "", fmt.Errorf("%s: marshal request: %w", c.provider, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.config.BaseURL+"/chat/completions", bytes.NewReader(requestBytes))
	if err != nil {
		return nil, "", fmt.Errorf("%s: build request: %w", c.provider, err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)

	start := time.Now()
	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("%s: http: %w", c.provider, err)
	}
	defer func() { _ = resp.Body.Close() }()

	metrics.RecordResponseLatency(metrics.LLMLatency, metrics.LLMLatencyLast,
		c.provider, c.config.Model, c.config.BaseURL, req, resp, start)

	proxyTotalMs := resp.Header.Get("x-proxy-total-ms")
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, proxyTotalMs, fmt.Errorf("%s: api %d: %s", c.provider, resp.StatusCode, body)
	}
	return body, proxyTotalMs, nil
}
