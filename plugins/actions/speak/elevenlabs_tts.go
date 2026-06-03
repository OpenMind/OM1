package speak

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers/tts"
)

type SpeakInput struct {
	Action string `json:"action" description:"The text to be spoken"`
}

func init() {
	actions.RegisterInterface(
		"speak",
		"Action interface for text-to-speech output. Makes the robot speak the provided text.",
		SpeakInput{},
	)
	actions.Register("speak/elevenlabs_tts", NewElevenLabsTTS)
}

type ElevenLabsConfig struct {
	APIKey           string `json:"api_key"`
	ElevenLabsAPIKey string `json:"elevenlabs_api_key"`
	VoiceID          string `json:"voice_id"`
	ModelID          string `json:"model_id"`
	OutputFormat     string `json:"output_format"`
	Rate             int    `json:"rate"`
	SilenceRate      int    `json:"silence_rate"`
}

type ElevenLabsConnector struct {
	elevenlabs *tts.ElevenLabsProvider
	log        *zap.Logger

	silenceMu      sync.Mutex
	silenceCounter int
	silenceRate    int
}

// NewElevenLabsTTS creates a new ElevenLabsConnector with the provided configuration.
func NewElevenLabsTTS(configMap map[string]any) (actions.Connector, error) {
	var cfg ElevenLabsConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("speak/elevenlabs_tts: api_key required")
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = tts.DefaultVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = tts.DefaultModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = tts.DefaultOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = rateFromFormat(cfg.OutputFormat)
	}

	log := logger.Get()
	elevenlabs := tts.ElevenLabs(tts.ElevenLabsConfig{
		APIKey:           cfg.APIKey,
		ElevenLabsAPIKey: cfg.ElevenLabsAPIKey,
		VoiceID:          cfg.VoiceID,
		ModelID:          cfg.ModelID,
		OutputFormat:     cfg.OutputFormat,
		Rate:             cfg.Rate,
	}, log)

	return &ElevenLabsConnector{
		elevenlabs:  elevenlabs,
		log:         log,
		silenceRate: cfg.SilenceRate,
	}, nil
}

// rateFromFormat extracts the sample rate from strings like "pcm_16000".
func rateFromFormat(format string) int {
	parts := strings.Split(format, "_")
	if len(parts) >= 2 {
		if r, err := strconv.Atoi(parts[len(parts)-1]); err == nil {
			return r
		}
	}
	return tts.DefaultRate
}

// Connect enqueues the spoken text without blocking the action pipeline.
func (e *ElevenLabsConnector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	arguments, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("speak/elevenlabs_tts: unexpected input type %T", input)
	}
	text, _ := arguments["action"].(string)
	if text == "" {
		return nil, nil
	}

	e.silenceMu.Lock()
	if e.silenceRate > 0 && e.silenceCounter < e.silenceRate {
		e.silenceCounter++
		e.silenceMu.Unlock()
		e.log.Info("speak/elevenlabs_tts: skipping (silence_rate)", zap.Int("counter", e.silenceCounter))
		return nil, nil
	}
	e.silenceCounter = 0
	e.silenceMu.Unlock()

	e.log.Info("speak/elevenlabs_tts: enqueueing text", zap.String("text", text))
	e.elevenlabs.AddText(text)

	return nil, nil
}

// Tick is a no-op since this connector is event-driven.
func (e *ElevenLabsConnector) Tick(ctx context.Context) {
	<-ctx.Done()
}

// Stop is a no-op since the ElevenLabs provider manages its own resources.
func (e *ElevenLabsConnector) Stop() {}
