package tts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"

	"go.uber.org/zap"
)

const (
	ElevenLabsTTSURL    = "https://api.openmind.com/api/core/elevenlabs/tts/audio/speech"
	DefaultVoiceID      = "JBFqnCBsd6RMkjVDRZzb"
	DefaultModelID      = "eleven_flash_v2_5"
	DefaultOutputFormat = "pcm_16000"
	DefaultRate         = 16000

	warmupSilenceMs = 1500
)

// ElevenLabsConfig holds connection and audio parameters for the provider.
type ElevenLabsConfig struct {
	APIKey           string
	ElevenLabsAPIKey string
	VoiceID          string
	ModelID          string
	OutputFormat     string
	Rate             int
}

var (
	elevenLabsOnce     sync.Once
	elevenLabsInstance *ElevenLabsProvider
)

// ElevenLabs returns the singleton ElevenLabsProvider instance, initializing it on first call.
func ElevenLabs(cfg ElevenLabsConfig, log *zap.Logger) *ElevenLabsProvider {
	elevenLabsOnce.Do(func() {
		elevenLabsInstance = newElevenLabsProvider(cfg, log)
		elevenLabsInstance.Start()
	})
	return elevenLabsInstance
}

// ElevenLabsProvider streams TTS audio from ElevenLabs through a persistent ffplay process.
type ElevenLabsProvider struct {
	*ttsBase
	cfg ElevenLabsConfig
}

func newElevenLabsProvider(cfg ElevenLabsConfig, log *zap.Logger) *ElevenLabsProvider {
	ctx, cancel := context.WithCancel(context.Background())
	p := &ElevenLabsProvider{
		ttsBase: &ttsBase{
			name:            "elevenlabs",
			rate:            cfg.Rate,
			outputFormat:    cfg.OutputFormat,
			log:             log,
			queue:           make(chan ttsRequest, providerQueueDepth),
			warmupSilenceMs: warmupSilenceMs,
			ctx:             ctx,
			cancel:          cancel,
		},
		cfg: cfg,
	}
	p.synth = p.synthesize
	return p
}

// synthesize posts text to the ElevenLabs endpoint and streams PCM chunks to ffplay.
func (p *ElevenLabsProvider) synthesize(req ttsRequest) error {
	voiceID := req.voiceID
	if voiceID == "" {
		voiceID = p.cfg.VoiceID
	}
	body := map[string]any{
		"model":           p.cfg.ModelID,
		"voice":           voiceID,
		"response_format": p.cfg.OutputFormat,
		"input":           req.text,
	}
	if p.cfg.ElevenLabsAPIKey != "" {
		body["elevenlabs_api_key"] = p.cfg.ElevenLabsAPIKey
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(p.ctx, http.MethodPost, ElevenLabsTTSURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.cfg.APIKey)

	return p.postAndStream(httpReq, p.cfg.ModelID, ElevenLabsTTSURL)
}
