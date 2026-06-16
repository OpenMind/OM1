package vlm

import (
	bg "github.com/openmind/om1/internal/backgrounds"
)

func init() {
	bg.Register("VLMGemini", NewVLMGemini)
	bg.Register("VLMGeminiRTSP", NewVLMGeminiRTSP)
}

var geminiDefaults = providerDefaults{
	baseURL: "https://api.openmind.com/api/core/gemini",
	model:   "gemini-2.5-flash",
	prompt: "In one concise sentence, describe what you see in this image. " +
		"Just the description — no explanation of your reasoning.",
	maxTokens: 1024,
}

// NewVLMGemini constructs a camera-backed Gemini VLM background.
func NewVLMGemini(configMap map[string]any) (bg.Background, error) {
	return NewCameraBackground("VLMGemini", geminiDefaults, configMap)
}

// NewVLMGeminiRTSP constructs an RTSP-backed Gemini VLM background.
func NewVLMGeminiRTSP(configMap map[string]any) (bg.Background, error) {
	return NewRTSPBackground("VLMGeminiRTSP", geminiDefaults, configMap)
}
