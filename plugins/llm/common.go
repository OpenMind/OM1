package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/metrics"
	"go.uber.org/zap"
)

// openAIResp models the structure of the response from OpenAI's API.
type openAIResp struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// parseOpenAIResponse converts the raw JSON response from OpenAI into the internal llm.Response format.
func parseOpenAIResponse(responseBytes []byte) (*llm.Response, error) {
	var apiResponse openAIResp
	if err := json.Unmarshal(responseBytes, &apiResponse); err != nil {
		return nil, err
	}
	if len(apiResponse.Choices) == 0 {
		return nil, fmt.Errorf("openai: empty choices")
	}

	firstChoice := apiResponse.Choices[0].Message
	response := &llm.Response{
		TextContent: firstChoice.Content,
		Usage: llm.Usage{
			PromptTokens:     apiResponse.Usage.PromptTokens,
			CompletionTokens: apiResponse.Usage.CompletionTokens,
		},
	}

	for _, toolCall := range firstChoice.ToolCalls {
		var arguments map[string]any
		_ = json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments)
		response.ToolCalls = append(response.ToolCalls, llm.ToolCall{
			Name:      toolCall.Function.Name,
			Arguments: arguments,
		})
	}
	return response, nil
}

// buildMessages constructs the message history in the format expected by the OpenAI API.
func buildMessages(prompt string, history []llm.Message) []map[string]any {
	messages := make([]map[string]any, 0, len(history)+1)
	for _, turn := range history {
		messages = append(messages, map[string]any{"role": turn.Role, "content": turn.Content})
	}
	messages = append(messages, map[string]any{"role": "user", "content": prompt})
	return messages
}

// recordResponseLatency logs the latency and relevant headers from the OpenAI
// API response and records the round-trip latency as Prometheus LLM metrics,
// labelled by the provider's model and endpoint.
func recordResponseLatency(provider, model, endpoint string, req *http.Request, resp *http.Response, start time.Time) {
	header := func(key string) string {
		if v := resp.Header.Get(key); v != "" {
			return v
		}
		return "?"
	}
	elapsed := time.Since(start)
	logger.Get().Info(provider,
		zap.String("method", req.Method),
		zap.String("url", req.URL.String()),
		zap.Int("status", resp.StatusCode),
		zap.Int64("elapsed_ms", elapsed.Milliseconds()),
		zap.String("proxy_parse_ms", header("x-proxy-parse-ms")),
		zap.String("upstream_total_ms", header("x-upstream-total-ms")),
		zap.String("upstream_ttfb_ms", header("x-upstream-ttfb-ms")),
		zap.String("proxy_total_ms", header("x-proxy-total-ms")),
	)

	seconds := elapsed.Seconds()
	metrics.LLMLatency.WithLabelValues(model, endpoint).Observe(seconds)
	metrics.LLMLatencyLast.WithLabelValues(model, endpoint).Set(seconds)

	metrics.RecordHTTPTiming(req.URL.Host, req.URL.Path, req.Method, resp.StatusCode,
		header("x-proxy-parse-ms"), header("x-upstream-total-ms"),
		header("x-upstream-ttfb-ms"), header("x-proxy-total-ms"))
}

// remarshal is a helper function that converts between two struct types by marshaling to JSON and then unmarshalling.
func remarshal(src any, dst any) error {
	jsonBytes, err := json.Marshal(src)
	if err != nil {
		return err
	}
	return json.Unmarshal(jsonBytes, dst)
}
