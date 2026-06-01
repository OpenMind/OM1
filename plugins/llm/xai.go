package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("XAILLM", NewXAI)
}

type xaiModel = string

const (
	XAIModelGrok2Latest xaiModel = "grok-2-latest"
	XAIModelGrok3Beta   xaiModel = "grok-3-beta"
	XAIModelGrok4Latest xaiModel = "grok-4-latest"
	XAIModelGrok4       xaiModel = "grok-4"

	defaultXAIModel   xaiModel = XAIModelGrok4Latest
	defaultXAIBaseURL string   = "https://api.openmind.com/api/core/xai"
)

// NewXAI creates an xAI (Grok) LLM using the OpenAI-compatible API.
func NewXAI(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("XAILLM", configMap, defaultXAIModel, defaultXAIBaseURL, "auto", true)
}
