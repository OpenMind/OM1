package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("NearAILLM", NewNearAI)
}

type nearAIModel = string

const (
	NearAIModelQwen30BA3BInstruct2507 nearAIModel = "qwen3-30b-a3b-instruct-2507"
	NearAIModelQwenVl72BInstruct      nearAIModel = "qwen2.5-vl-72b-instruct"
	NearAIModelQwen7BInstruct         nearAIModel = "qwen2.5-7b-instruct"

	defaultNearAIModel   nearAIModel = NearAIModelQwen30BA3BInstruct2507
	defaultNearAIBaseURL string      = "https://api.openmind.com/api/core/nearai"
)

func NewNearAI(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("NearAILLM", configMap, defaultNearAIModel, defaultNearAIBaseURL, "auto", true)
}
