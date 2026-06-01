package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("QwenLLM", NewQwen)
}

const (
	defaultQwenModel   string = "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"
	defaultQwenBaseURL string = "http://127.0.0.1:8860/v1"
	defaultQwenAPIKey  string = "placeholder"
)

// qwenToolCallRe matches Qwen-style inline tool call blocks emitted as text.
var qwenToolCallRe = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

// qwenLLM is a local Qwen model served over an OpenAI-compatible API. It differs
// from a plain provider in three ways: it disables the model's thinking mode via
// a "/no_think" suffix and chat_template_kwargs, and it falls back to parsing
// tool calls emitted as <tool_call> text when the server returns none structured.
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

// Call sends a prompt to the local Qwen model, appending "/no_think" when
// reasoning is disabled and recovering tool calls from text when needed.
func (q *qwenLLM) Call(ctx context.Context, prompt string, history []llm.Message) (*llm.Response, error) {
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

	body, err := q.doRequest(ctx, requestBody)
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
	return resp, nil
}

// parseQwenToolCalls extracts <tool_call>{...}</tool_call> blocks from text.
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
