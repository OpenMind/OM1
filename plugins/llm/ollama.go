package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("OllamaLLM", NewOllama)
}

const (
	defaultOllamaModel   string  = "llama3.2"
	defaultOllamaBaseURL string  = "http://localhost:11434"
	defaultOllamaTemp    float64 = 0.7
	defaultOllamaNumCtx  int     = 4096
)

type ollamaConfig struct {
	BaseURL     string  `json:"base_url"`
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
	NumCtx      int     `json:"num_ctx"`
}

// ollamaLLM talks to a local Ollama server over its native /api/chat protocol,
// providing privacy-focused, offline-capable inference.
type ollamaLLM struct {
	config  ollamaConfig
	chatURL string
	schemas []map[string]any
}

// NewOllama creates an Ollama LLM backed by a local Ollama server.
func NewOllama(configMap map[string]any) (llm.LLM, error) {
	cfg := ollamaConfig{Temperature: defaultOllamaTemp, NumCtx: defaultOllamaNumCtx}
	if err := remarshal(configMap, &cfg); err != nil {
		return nil, fmt.Errorf("OllamaLLM config: %w", err)
	}
	if cfg.Model == "" {
		cfg.Model = defaultOllamaModel
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultOllamaBaseURL
	}
	baseURL := strings.Trim(cfg.BaseURL, "/")
	return &ollamaLLM{config: cfg, chatURL: baseURL + "/api/chat"}, nil
}

// FunctionSchemas returns the current function schemas registered with the LLM.
func (o *ollamaLLM) FunctionSchemas() []map[string]any { return o.schemas }

// SetSchemas updates the function schemas that the LLM will use for tool calls.
func (o *ollamaLLM) SetSchemas(schemas []map[string]any) { o.schemas = schemas }

// ollamaResp models the structure of the response from Ollama's /api/chat endpoint.
type ollamaResp struct {
	Message struct {
		Content   string `json:"content"`
		ToolCalls []struct {
			Function struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"message"`
	PromptEvalCount int `json:"prompt_eval_count"`
	EvalCount       int `json:"eval_count"`
}

// Call sends a prompt and conversation history to the Ollama server and returns the response.
func (o *ollamaLLM) Call(ctx context.Context, prompt string, history []llm.Message) (*llm.Response, error) {
	requestBody := map[string]any{
		"model":    o.config.Model,
		"messages": buildMessages(prompt, history),
		"stream":   false,
		"options": map[string]any{
			"temperature": o.config.Temperature,
			"num_ctx":     o.config.NumCtx,
		},
	}
	if len(o.schemas) > 0 {
		requestBody["tools"] = o.schemas
	}

	requestBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("OllamaLLM: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, o.chatURL, bytes.NewReader(requestBytes))
	if err != nil {
		return nil, fmt.Errorf("OllamaLLM: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return nil, fmt.Errorf("OllamaLLM: http (is Ollama running at %s?): %w", o.chatURL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	logResponseLatency("OllamaLLM", req, resp, start)

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OllamaLLM: api %d: %s", resp.StatusCode, body)
	}

	var apiResponse ollamaResp
	if err := json.Unmarshal(body, &apiResponse); err != nil {
		return nil, fmt.Errorf("OllamaLLM: parse response: %w", err)
	}

	response := &llm.Response{
		TextContent: apiResponse.Message.Content,
		Usage: llm.Usage{
			PromptTokens:     apiResponse.PromptEvalCount,
			CompletionTokens: apiResponse.EvalCount,
		},
	}
	for _, toolCall := range apiResponse.Message.ToolCalls {
		response.ToolCalls = append(response.ToolCalls, llm.ToolCall{
			Name:      toolCall.Function.Name,
			Arguments: toolCall.Function.Arguments,
		})
	}
	return response, nil
}
