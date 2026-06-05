package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("FunctionGemmaLLM", NewFunctionGemma)
}

type functionGemmaModel = string

const (
	FunctionGemmaModelMultilingual functionGemmaModel = "functiongemma-finetuned-g1-multilingual"

	defaultFunctionGemmaModel   functionGemmaModel = FunctionGemmaModelMultilingual
	defaultFunctionGemmaBaseURL string             = "http://localhost:8200/v1"
)

func NewFunctionGemma(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("FunctionGemmaLLM", configMap, defaultFunctionGemmaModel, defaultFunctionGemmaBaseURL, "auto", true)
}
