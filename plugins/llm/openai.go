package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("OpenAILLM", NewOpenAI)
}

type openAIModel = string

const (
	OpenAIModelGPT4o     openAIModel = "gpt-4o"
	OpenAIModelGPT4oMini openAIModel = "gpt-4o-mini"
	OpenAIModelGPT41     openAIModel = "gpt-4.1"
	OpenAIModelGPT41Mini openAIModel = "gpt-4.1-mini"
	OpenAIModelGPT41Nano openAIModel = "gpt-4.1-nano"
	OpenAIModelGPT5      openAIModel = "gpt-5"
	OpenAIModelGPT5Mini  openAIModel = "gpt-5-mini"
	OpenAIModelGPT5Nano  openAIModel = "gpt-5-nano"
	OpenAIModelGPT51     openAIModel = "gpt-5.1"
	OpenAIModelGPT52     openAIModel = "gpt-5.2"

	defaultOpenAIModel   openAIModel = OpenAIModelGPT41Mini
	defaultOpenAIBaseURL string      = "https://api.openmind.com/api/core/openai"
)

func NewOpenAI(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("OpenAILLM", configMap, defaultOpenAIModel, defaultOpenAIBaseURL, "auto", true)
}
