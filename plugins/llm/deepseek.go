package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("DeepSeekLLM", NewDeepSeek)
}

type deepSeekModel = string

const (
	DeepSeekModelChat deepSeekModel = "deepseek-chat"

	defaultDeepSeekModel   deepSeekModel = DeepSeekModelChat
	defaultDeepSeekBaseURL string        = "https://api.openmind.com/api/core/deepseek"
)

func NewDeepSeek(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("DeepSeekLLM", configMap, defaultDeepSeekModel, defaultDeepSeekBaseURL, "auto", true)
}
