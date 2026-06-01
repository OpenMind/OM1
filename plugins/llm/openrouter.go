package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("OpenRouter", NewOpenRouter)
}

type openRouterModel = string

const (
	OpenRouterModelAnthropicSonnet45 openRouterModel = "anthropic/claude-sonnet-4.5"
	OpenRouterModelAnthropicOpus45   openRouterModel = "anthropic/claude-opus-4.5"
	OpenRouterModelAnthropicHaiku45  openRouterModel = "anthropic/claude-haiku-4.5"
	OpenRouterModelMoonshotKimiK25   openRouterModel = "moonshotai/kimi-k2.5"
	OpenRouterModelMinimaxM21        openRouterModel = "minimax/minimax-m2.1"
	OpenRouterModelZAIGLM47          openRouterModel = "z-ai/glm-4.7"
	OpenRouterModelXAIGrok4Fast      openRouterModel = "x-ai/grok-4-fast"
	OpenRouterModelDeepSeekV32       openRouterModel = "deepseek/deepseek-v3.2"
	OpenRouterModelLlama3370B        openRouterModel = "meta-llama/llama-3.3-70b-instruct"

	defaultOpenRouterModel   openRouterModel = OpenRouterModelAnthropicSonnet45
	defaultOpenRouterBaseURL string          = "https://api.openmind.com/api/core/openrouter"
)

// NewOpenRouter creates an OpenRouter LLM using the OpenAI-compatible API.
func NewOpenRouter(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("OpenRouter", configMap, defaultOpenRouterModel, defaultOpenRouterBaseURL, "auto", true)
}
