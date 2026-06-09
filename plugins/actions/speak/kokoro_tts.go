package speak

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers/tts"
)

func init() {
	actions.Register("speak/kokoro_tts", NewKokoroTTS)
}

// KokoroConfig is the JSON configuration for the Kokoro TTS connector.
type KokoroConfig struct {
	APIKey       string `json:"api_key"`
	BaseURL      string `json:"base_url"`
	VoiceID      string `json:"voice_id"`
	ModelID      string `json:"model_id"`
	OutputFormat string `json:"output_format"`
	Rate         int    `json:"rate"`
	SilenceRate  int    `json:"silence_rate"`
}

// KokoroConnector is a "speak" connector that streams Text-to-Speech audio from a
// Kokoro (OpenAI-compatible) TTS endpoint.
type KokoroConnector struct {
	kokoro *tts.KokoroProvider
	log    *zap.Logger

	silenceMu      sync.Mutex
	silenceCounter int
	silenceRate    int
}

// NewKokoroTTS creates a new KokoroConnector with the provided configuration.
func NewKokoroTTS(configMap map[string]any) (actions.Connector, error) {
	var cfg KokoroConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = tts.DefaultKokoroBaseURL
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = tts.DefaultKokoroVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = tts.DefaultKokoroModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = tts.DefaultKokoroOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = tts.DefaultKokoroRate
	}

	log := logger.Get().Named("speak/kokoro_tts")
	kokoro := tts.Kokoro(tts.KokoroConfig{
		BaseURL:      cfg.BaseURL,
		APIKey:       cfg.APIKey,
		VoiceID:      cfg.VoiceID,
		ModelID:      cfg.ModelID,
		OutputFormat: cfg.OutputFormat,
		Rate:         cfg.Rate,
	}, log)

	return &KokoroConnector{
		kokoro:      kokoro,
		log:         log,
		silenceRate: cfg.SilenceRate,
	}, nil
}

// Connect enqueues the spoken text without blocking the action pipeline.
func (k *KokoroConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	arguments, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("speak/kokoro_tts: unexpected input type %T", input)
	}
	text, _ := arguments["action"].(string)
	if text == "" {
		return nil, nil
	}

	k.silenceMu.Lock()
	if k.silenceRate > 0 && k.silenceCounter < k.silenceRate {
		k.silenceCounter++
		k.silenceMu.Unlock()
		k.log.Info("skipping (silence_rate)", zap.Int("counter", k.silenceCounter))
		return nil, nil
	}
	k.silenceCounter = 0
	k.silenceMu.Unlock()

	k.log.Info("enqueueing text", zap.String("text", text))
	k.kokoro.AddText(text)

	return nil, nil
}

// Tick is a no-op since this connector is event-driven.
func (k *KokoroConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since the Kokoro provider manages its own resources.
func (k *KokoroConnector) Stop() {}
