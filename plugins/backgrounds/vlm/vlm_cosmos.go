package vlm

import (
	"time"

	bg "github.com/openmind/om1/internal/backgrounds"
)

func init() {
	bg.Register("VLMCosmos", NewVLMCosmos)
	bg.Register("VLMCosmosRTSP", NewVLMCosmosRTSP)
}

var cosmosDefaults = providerDefaults{
	baseURL: "http://127.0.0.1:8000/v1",
	apiKey:  "placeholder",
	model:   "nvidia/Cosmos3-Edge",
	prompt: "In one short sentence, state how many people are visible, what they are doing, " +
		"and any readable text. No other words.",
	maxTokens: 64,
	extraBody: map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}},
	timeout:   10 * time.Second,
	warmup:    true,
}

// NewVLMCosmos constructs a camera-backed Cosmos VLM background.
func NewVLMCosmos(configMap map[string]any) (bg.Background, error) {
	return NewCameraBackground("VLMCosmos", cosmosDefaults, configMap)
}

// NewVLMCosmosRTSP constructs an RTSP-backed Cosmos VLM background.
func NewVLMCosmosRTSP(configMap map[string]any) (bg.Background, error) {
	return NewRTSPBackground("VLMCosmosRTSP", cosmosDefaults, configMap)
}
