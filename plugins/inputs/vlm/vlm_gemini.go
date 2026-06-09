package vlm

import (
	"github.com/openmind/om1/internal/inputs"
)

func init() {
	inputs.Register("VLMGemini", NewVLMGemini)
	inputs.Register("VLMGeminiRTSP", NewVLMGeminiRTSP)
}

var geminiDefaults = providerDefaults{
	baseURL: "https://api.openmind.com/api/core/gemini",
	model:   "gemini-2.5-flash",
	prompt: "In one concise sentence, describe what you see in this image. " +
		"Just the description — no explanation of your reasoning.",
	maxTokens: 1024,
}

func NewVLMGemini(configMap map[string]any) (inputs.Sensor, error) {
	return NewCameraSensor("VLMGemini", geminiDefaults, configMap)
}

func NewVLMGeminiRTSP(configMap map[string]any) (inputs.Sensor, error) {
	return NewRTSPSensor("VLMGeminiRTSP", geminiDefaults, configMap)
}
