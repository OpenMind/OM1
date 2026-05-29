package llm

import (
	"encoding/json"
	"fmt"

	"github.com/openmind/om1/internal/llm"
)

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

func buildMessages(prompt string, history []llm.Message) []map[string]any {
	messages := make([]map[string]any, 0, len(history)+1)
	for _, turn := range history {
		messages = append(messages, map[string]any{"role": turn.Role, "content": turn.Content})
	}
	messages = append(messages, map[string]any{"role": "user", "content": prompt})
	return messages
}

func remarshal(src any, dst any) error {
	jsonBytes, err := json.Marshal(src)
	if err != nil {
		return err
	}
	return json.Unmarshal(jsonBytes, dst)
}
