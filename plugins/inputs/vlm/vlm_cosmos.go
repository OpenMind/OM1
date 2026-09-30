package vlm

import (
	"time"

	"github.com/openmind/om1/internal/inputs"
)

func init() {
	inputs.Register("VLMCosmos", NewVLMCosmos)
	inputs.Register("VLMCosmosRTSP", NewVLMCosmosRTSP)
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

func NewVLMCosmos(configMap map[string]any) (inputs.Sensor, error) {
	return NewCameraSensor("VLMCosmos", cosmosDefaults, configMap)
}

func NewVLMCosmosRTSP(configMap map[string]any) (inputs.Sensor, error) {
	return NewRTSPSensor("VLMCosmosRTSP", cosmosDefaults, configMap)
}
