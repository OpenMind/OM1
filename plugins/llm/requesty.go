package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("Requesty", NewRequesty)
}

type requestyModel = string

const (
	RequestyModelAnthropicSonnet45 requestyModel = "anthropic/claude-sonnet-4-5"
	RequestyModelOpenAIGPT4oMini   requestyModel = "openai/gpt-4o-mini"
	RequestyModelGeminiFlash25     requestyModel = "google/gemini-2.5-flash"
	RequestyModelDeepSeekChat      requestyModel = "deepseek/deepseek-chat"
	RequestyModelXAIGrok4Fast      requestyModel = "x-ai/grok-4-fast"

	defaultRequestyModel   requestyModel = RequestyModelAnthropicSonnet45
	defaultRequestyBaseURL string        = "https://router.requesty.ai/v1"
)

func NewRequesty(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("Requesty", configMap, defaultRequestyModel, defaultRequestyBaseURL, "auto", true)
}
