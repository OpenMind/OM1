package luma_checkin

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers/tts"
)

const (
	defaultGreetingTemplate = "Welcome, {first_name}! Glad you made it."
	defaultStatus           = "checked_in"
	sideEffectTimeout       = 5 * time.Second
)

// LumaCheckinInput is the LLM-visible argument schema. The event is configured
// per-deployment; the LLM only supplies the pk parsed from the QR code.
type LumaCheckinInput struct {
	Pk string `json:"pk" description:"The Luma guest or ticket key (pk parameter from check-in QR code)"`
}

// ttsConfig configures which TTS provider to use for the welcome utterance.
// Both Kokoro and ElevenLabs providers are sync.Once singletons, so this
// instance is shared with the speak action plugin if both are configured;
// the first caller's config wins.
type ttsConfig struct {
	Provider         string `json:"provider"`
	APIKey           string `json:"api_key"`
	ElevenLabsAPIKey string `json:"elevenlabs_api_key"`
	BaseURL          string `json:"base_url"`
	VoiceID          string `json:"voice_id"`
	ModelID          string `json:"model_id"`
	OutputFormat     string `json:"output_format"`
	Rate             int    `json:"rate"`
}

// Config holds the plugin configuration decoded from JSON5.
type Config struct {
	APIKey                string    `json:"api_key"`
	BaseURL               string    `json:"base_url"`
	EventAPIID            string    `json:"event_api_id"`
	GreetingTemplate      string    `json:"greeting_template"`
	RequestTimeoutSeconds float64   `json:"request_timeout_seconds"`
	StatusValue           string    `json:"status_value"`
	TTS                   ttsConfig `json:"tts"`
}

// ttsPlayer is the minimal interface luma_checkin needs from a TTS provider.
type ttsPlayer interface {
	AddText(text string)
}

// guestLookup is the interface luma_checkin uses against Luma; abstracted for tests.
type guestLookup interface {
	GetGuest(ctx context.Context, pk string) (*Guest, error)
	UpdateGuestStatus(ctx context.Context, eventAPIID, guestAPIID, status string) error
}

type connector struct {
	log    *zap.Logger
	cfg    Config
	client guestLookup
	tts    ttsPlayer

	mu      sync.Mutex
	stopped bool
}

func init() {
	actions.RegisterInterface(
		"luma_checkin",
		"Action interface for checking guests into a Luma event. "+
			"Looks up the guest via the Luma external check-in API, validates the configured event, "+
			"flips status to checked_in, and speaks a personalized welcome via the configured TTS provider. "+
			"For arm gestures and facial expressions, call unitree_g1_arm and emotion alongside this action.",
		LumaCheckinInput{},
	)
	actions.Register("luma_checkin/api", NewLumaCheckinConnector)
}

// NewLumaCheckinConnector constructs a connector from the decoded config map.
func NewLumaCheckinConnector(cfgMap map[string]any) (actions.Connector, error) {
	cfg, err := parseConfig(cfgMap)
	if err != nil {
		return nil, err
	}
	log := logger.Get().Named("luma_checkin/api")

	timeout := time.Duration(cfg.RequestTimeoutSeconds * float64(time.Second))
	c := &connector{
		log:    log,
		cfg:    cfg,
		client: newLumaClient(cfg.BaseURL, cfg.APIKey, cfg.EventAPIID, timeout),
		tts:    buildTTS(cfg.TTS, log),
	}

	log.Info("initialized",
		zap.String("event_api_id", cfg.EventAPIID),
		zap.String("tts_provider", cfg.TTS.Provider),
	)
	return c, nil
}

func parseConfig(cfgMap map[string]any) (Config, error) {
	var cfg Config
	if b, err := json.Marshal(cfgMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return cfg, fmt.Errorf("luma_checkin: api_key is required")
	}
	if cfg.EventAPIID == "" {
		return cfg, fmt.Errorf("luma_checkin: event_api_id is required")
	}
	if cfg.GreetingTemplate == "" {
		cfg.GreetingTemplate = defaultGreetingTemplate
	}
	if cfg.StatusValue == "" {
		cfg.StatusValue = defaultStatus
	}
	if cfg.RequestTimeoutSeconds <= 0 {
		cfg.RequestTimeoutSeconds = 5
	}
	return cfg, nil
}

func buildTTS(cfg ttsConfig, log *zap.Logger) ttsPlayer {
	switch strings.ToLower(cfg.Provider) {
	case "kokoro":
		return tts.Kokoro(tts.KokoroConfig{
			BaseURL:      orDefault(cfg.BaseURL, tts.DefaultKokoroBaseURL),
			APIKey:       cfg.APIKey,
			VoiceID:      orDefault(cfg.VoiceID, tts.DefaultKokoroVoiceID),
			ModelID:      orDefault(cfg.ModelID, tts.DefaultKokoroModelID),
			OutputFormat: orDefault(cfg.OutputFormat, tts.DefaultKokoroOutputFormat),
			Rate:         orDefaultInt(cfg.Rate, tts.DefaultKokoroRate),
		}, log)
	case "elevenlabs":
		return tts.ElevenLabs(tts.ElevenLabsConfig{
			APIKey:           cfg.APIKey,
			ElevenLabsAPIKey: cfg.ElevenLabsAPIKey,
			VoiceID:          orDefault(cfg.VoiceID, tts.DefaultVoiceID),
			ModelID:          orDefault(cfg.ModelID, tts.DefaultModelID),
			OutputFormat:     orDefault(cfg.OutputFormat, tts.DefaultOutputFormat),
			Rate:             orDefaultInt(cfg.Rate, tts.DefaultRate),
		}, log)
	case "":
		return nil
	default:
		log.Warn("luma_checkin: unknown tts.provider", zap.String("provider", cfg.Provider))
		return nil
	}
}

// Connect performs the check-in: GetGuest → validate event → fan out the
// status update + TTS welcome → return status string. Arm wave and emotion
// are intentionally not handled here; configure unitree_g1_arm and emotion
// actions and let the LLM call them concurrently.
func (c *connector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("luma_checkin: unexpected input type %T", input)
	}
	pk := strings.TrimSpace(stringField(args, "pk"))
	if pk == "" {
		return nil, fmt.Errorf("luma_checkin: pk required")
	}

	guest, err := c.client.GetGuest(ctx, pk)
	if err != nil {
		switch {
		case errors.Is(err, errLumaNotFound):
			c.log.Info("guest not found", zap.String("pk", pk))
			return "checkin_failed: guest not registered for this event", nil
		case errors.Is(err, errLumaUnauthorized):
			c.log.Error("luma auth failed")
			return nil, fmt.Errorf("luma api auth failed")
		default:
			c.log.Warn("get-guest failed", zap.Error(err))
			return nil, err
		}
	}
	if guest == nil {
		return "checkin_failed: empty guest response", nil
	}
	if guest.EventAPIID != "" && guest.EventAPIID != c.cfg.EventAPIID {
		c.log.Info("event mismatch",
			zap.String("expected", c.cfg.EventAPIID),
			zap.String("got", guest.EventAPIID),
		)
		return "checkin_failed: qr is for a different event", nil
	}

	displayName := firstNameFor(guest)

	sideCtx, cancel := context.WithTimeout(ctx, sideEffectTimeout)
	defer cancel()

	var wg sync.WaitGroup
	var statusErr error
	var statusMu sync.Mutex

	if guest.APIID != "" {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := c.client.UpdateGuestStatus(sideCtx, c.cfg.EventAPIID, guest.APIID, c.cfg.StatusValue); err != nil {
				c.log.Warn("update-guest-status failed", zap.Error(err))
				statusMu.Lock()
				statusErr = err
				statusMu.Unlock()
			}
		}()
	}

	if c.tts != nil {
		c.tts.AddText(formatGreeting(c.cfg.GreetingTemplate, guest))
	}

	wg.Wait()

	statusMu.Lock()
	failed := statusErr != nil
	statusMu.Unlock()

	c.log.Info("checked in", zap.String("pk", pk), zap.String("name", displayName), zap.Bool("status_update_failed", failed))
	if failed {
		return fmt.Sprintf("checked_in: %s (status update failed, logged)", displayName), nil
	}
	return fmt.Sprintf("checked_in: %s", displayName), nil
}

func (c *connector) Tick(ctx context.Context) {
	<-ctx.Done()
}

func (c *connector) Stop() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stopped = true
}

func stringField(m map[string]any, key string) string {
	v, _ := m[key].(string)
	return v
}

func firstNameFor(g *Guest) string {
	if g.FirstName != "" {
		return g.FirstName
	}
	if g.Name != "" {
		if parts := strings.Fields(g.Name); len(parts) > 0 {
			return parts[0]
		}
	}
	return "friend"
}

func formatGreeting(template string, g *Guest) string {
	r := strings.NewReplacer(
		"{first_name}", firstNameFor(g),
		"{last_name}", g.LastName,
		"{name}", orDefault(g.Name, firstNameFor(g)),
		"{email}", g.Email,
	)
	return r.Replace(template)
}

func orDefault(v, def string) string {
	if v != "" {
		return v
	}
	return def
}

func orDefaultInt(v, def int) int {
	if v > 0 {
		return v
	}
	return def
}
