package tts

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"go.uber.org/zap"
)

const (
	DefaultKokoroBaseURL      = "http://127.0.0.1:8880/v1"
	DefaultKokoroVoiceID      = "af_bella"
	DefaultKokoroModelID      = "kokoro"
	DefaultKokoroOutputFormat = "pcm"
	DefaultKokoroRate         = 24000

	kokoroSpeechPath = "/audio/speech"

	kokoroPreRollMs = 10
)

// KokoroConfig holds connection and audio parameters for the Kokoro provider.
type KokoroConfig struct {
	BaseURL      string
	APIKey       string
	VoiceID      string
	ModelID      string
	OutputFormat string
	Rate         int
}

var (
	kokoroOnce     sync.Once
	kokoroInstance *KokoroProvider
)

// Kokoro returns the singleton KokoroProvider instance, initializing it on first call.
func Kokoro(cfg KokoroConfig, log *zap.Logger) *KokoroProvider {
	kokoroOnce.Do(func() {
		kokoroInstance = newKokoroProvider(cfg, log)
		kokoroInstance.Start()
	})
	return kokoroInstance
}

// KokoroProvider streams TTS audio from a Kokoro (OpenAI-compatible) endpoint
// through a persistent ffplay process.
type KokoroProvider struct {
	*ttsBase
	cfg       KokoroConfig
	speechURL string
}

func newKokoroProvider(cfg KokoroConfig, log *zap.Logger) *KokoroProvider {
	ctx, cancel := context.WithCancel(context.Background())
	p := &KokoroProvider{
		ttsBase: &ttsBase{
			name:             "kokoro",
			rate:             cfg.Rate,
			outputFormat:     cfg.OutputFormat,
			log:              log,
			queue:            make(chan ttsRequest, providerQueueDepth),
			preRollSilenceMs: kokoroPreRollMs,
			ctx:              ctx,
			cancel:           cancel,
		},
		cfg:       cfg,
		speechURL: strings.TrimRight(cfg.BaseURL, "/") + kokoroSpeechPath,
	}
	p.synth = p.synthesize
	return p
}

// synthesize posts text to the Kokoro endpoint and streams PCM chunks to ffplay.
func (p *KokoroProvider) synthesize(req ttsRequest) error {
	body := map[string]any{
		"model":           p.cfg.ModelID,
		"voice":           p.cfg.VoiceID,
		"response_format": p.cfg.OutputFormat,
		"input":           req.text,
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(p.ctx, http.MethodPost, p.speechURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.cfg.APIKey != "" {
		httpReq.Header.Set("x-api-key", p.cfg.APIKey)
	}

	return p.postAndStream(httpReq, p.cfg.ModelID, p.speechURL)
}
