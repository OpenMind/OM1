package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/metrics"
)

func init() {
	llm.Register("QwenLLM", NewQwen)
}

const (
	defaultQwenModel   string = "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"
	defaultQwenBaseURL string = "http://127.0.0.1:8860/v1"
	defaultQwenAPIKey  string = "placeholder"
)

var qwenToolCallRe = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

type qwenLLM struct {
	*openAICompatLLM
	enableReasoning bool
}

// NewQwen creates a local Qwen LLM using the OpenAI-compatible API.
func NewQwen(configMap map[string]any) (llm.LLM, error) {
	base, err := newOpenAICompat("QwenLLM", configMap, defaultQwenModel, defaultQwenBaseURL, "required", false)
	if err != nil {
		return nil, err
	}
	if base.config.APIKey == "" {
		base.config.APIKey = defaultQwenAPIKey
	}
	base.extraBody = map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	}

	var opts struct {
		EnableReasoning bool `json:"enable_reasoning"`
	}
	if err := remarshal(configMap, &opts); err != nil {
		return nil, fmt.Errorf("QwenLLM config: %w", err)
	}

	return &qwenLLM{openAICompatLLM: base, enableReasoning: opts.EnableReasoning}, nil
}

func (q *qwenLLM) Call(ctx context.Context, prompt string, history []llm.Message) (*llm.Response, error) {
	// start brackets the whole operation: build + travel + proxy + parse.
	start := time.Now()

	userPrompt := prompt
	if !q.enableReasoning {
		userPrompt = prompt + " /no_think"
	}

	requestBody := map[string]any{
		"model":    q.config.Model,
		"messages": buildMessages(userPrompt, history),
	}
	if len(q.schemas) > 0 {
		requestBody["tools"] = q.schemas
		requestBody["tool_choice"] = q.toolChoice
	}
	for k, v := range q.extraBody {
		requestBody[k] = v
	}

	body, proxyTotalMs, err := q.doRequest(ctx, requestBody)
	if err != nil {
		return nil, err
	}

	resp, err := parseOpenAIResponse(body)
	if err != nil {
		return nil, err
	}

	if len(resp.ToolCalls) == 0 && strings.Contains(resp.TextContent, "<tool_call>") {
		resp.ToolCalls = parseQwenToolCalls(resp.TextContent)
	}
	metrics.RecordRequestTiming("llm", time.Since(start).Seconds(), proxyTotalMs)
	return resp, nil
}

func parseQwenToolCalls(text string) []llm.ToolCall {
	var toolCalls []llm.ToolCall
	for _, match := range qwenToolCallRe.FindAllStringSubmatch(text, -1) {
		var obj struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(match[1]), &obj); err != nil {
			continue
		}
		if obj.Name == "" {
			continue
		}
		toolCalls = append(toolCalls, llm.ToolCall{Name: obj.Name, Arguments: obj.Arguments})
	}
	return toolCalls
}
