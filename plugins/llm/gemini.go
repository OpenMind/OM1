package llm

import (
	"github.com/openmind/om1/internal/llm"
)

func init() {
	llm.Register("GeminiLLM", NewGemini)
}

type geminiModel = string

const (
	GeminiModel25Flash      geminiModel = "gemini-2.5-flash"
	GeminiModel25FlashLite  geminiModel = "gemini-2.5-flash-lite"
	GeminiModel25Pro        geminiModel = "gemini-2.5-pro"
	GeminiModel31ProPreview geminiModel = "gemini-3.1-pro-preview"
	GeminiModel31FlashLite  geminiModel = "gemini-3.1-flash-lite"
	GeminiModel35Flash      geminiModel = "gemini-3.5-flash"

	defaultGeminiModel   geminiModel = GeminiModel31FlashLite
	defaultGeminiBaseURL string      = "https://api.openmind.com/api/core/gemini"
)

func NewGemini(configMap map[string]any) (llm.LLM, error) {
	return newOpenAICompat("GeminiLLM", configMap, defaultGeminiModel, defaultGeminiBaseURL, "required", true)
}
