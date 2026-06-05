// Package forget_last implements the undo-last-enrollment action.
//
// Calls /gallery/forget_last on the face API. Soft-deletes the UUID
// that was created or claimed by the most recent /selfie within ~60s
// (API enforces the TTL). Used when the WRONG PERSON was captured:
// someone walked in front of the camera, or the user wasn't actually
// facing it.
//
// NOT for typos (use correct_identity) and NOT for look-alikes (use
// selfie with force=true).
//
// LLM workflow
// ------------
// The action takes NO arguments — the API server knows which UUID is
// the "last enrollment" via its internal last_enrollment state, set
// by the most recent successful /selfie call. The LLM doesn't need to
// supply any identifier.
//
// Reversibility
// -------------
// The forgotten UUID is moved to _trash/, not hard-deleted. The
// /gallery/restore endpoint (without arguments) restores the most
// recently soft-deleted UUID. Not currently exposed to the LLM —
// operator-level recovery only.
package forget_last

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
)

// ForgetLastInput — empty schema. Action takes no LLM-facing arguments.
// The server identifies the "last enrollment" from its own state
// (set by the most recent /selfie call).
type ForgetLastInput struct{}

func init() {
	actions.RegisterInterface(
		"forget_last",
		"Undo the most recent /selfie enrollment by soft-deleting its UUID. "+
			"Use when SelfieStatus showed result=success or result=merged within "+
			"~60s AND the user indicates the WRONG PERSON was captured (e.g. "+
			"'someone walked in front', 'that wasn't me'). NOT for typos (use "+
			"correct_identity). NOT for look-alikes (use selfie with force=true). "+
			"Takes no arguments — the server already knows which enrollment was "+
			"last.",
		ForgetLastInput{},
	)
	actions.Register("forget_last", NewHTTPConnector)
}

type Config struct {
	APIKey           string  `json:"api_key"`
	ElevenLabsAPIKey string  `json:"elevenlabs_api_key"`
	VoiceID          string  `json:"voice_id"`
	ModelID          string  `json:"model_id"`
	OutputFormat     string  `json:"output_format"`
	Rate             int     `json:"rate"`
	FaceHTTPBaseURL  string  `json:"face_http_base_url"`
	HTTPTimeoutSec   float64 `json:"http_timeout_sec"`
}

type Connector struct {
	log    *zap.Logger
	cfg    Config
	client *http.Client
	tts    *tts.ElevenLabsProvider
}

func NewHTTPConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("forget_last/http: api_key required")
	}
	if cfg.FaceHTTPBaseURL == "" {
		cfg.FaceHTTPBaseURL = "http://127.0.0.1:6793"
	}
	if cfg.HTTPTimeoutSec == 0 {
		cfg.HTTPTimeoutSec = 5.0
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
		cfg.Rate = tts.DefaultRate
	}

	log := logger.Get()
	ttsClient := tts.ElevenLabs(tts.ElevenLabsConfig{
		APIKey:           cfg.APIKey,
		ElevenLabsAPIKey: cfg.ElevenLabsAPIKey,
		VoiceID:          cfg.VoiceID,
		ModelID:          cfg.ModelID,
		OutputFormat:     cfg.OutputFormat,
		Rate:             cfg.Rate,
	}, log)

	return &Connector{
		log:    log,
		cfg:    cfg,
		client: &http.Client{Timeout: time.Duration(cfg.HTTPTimeoutSec * float64(time.Second))},
		tts:    ttsClient,
	}, nil
}

func (c *Connector) Connect(_ context.Context, _ actions.Input) (actions.Output, error) {
	// No arguments — empty body. The server identifies the target via
	// its own last_enrollment state.
	resp := c.postJSON("/gallery/forget_last", map[string]any{})
	c.dispatchResponse(resp)
	return nil, nil
}

func (c *Connector) Tick(ctx context.Context) { <-ctx.Done() }
func (c *Connector) Stop()                    {}

func (c *Connector) postJSON(path string, body map[string]any) map[string]any {
	url := c.cfg.FaceHTTPBaseURL + path
	buf, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, bytes.NewReader(buf))
	if err != nil {
		c.log.Warn("forget_last/http: build request failed", zap.Error(err))
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.log.Warn("forget_last/http: HTTP error", zap.Error(err))
		return nil
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		c.log.Warn("forget_last/http: decode failed", zap.Error(err))
		return nil
	}
	return out
}

// writeStatus surfaces a SelfieStatus line to the LLM. All face-memory
// actions write to the same key — the LLM disambiguates by reading
// the `result=...` prefix.
func (c *Connector) writeStatus(line string) {
	providers.IO().AddInput("SelfieStatus", line, time.Time{})
	c.log.Info("forget_last/http: status", zap.String("line", line))
}

func (c *Connector) speak(message string) {
	if message == "" {
		return
	}
	c.tts.AddText(message)
}

// shortUUID returns the first 8 chars of a UUID for compact logging.
func shortUUID(uuid string) string {
	if len(uuid) >= 8 {
		return uuid[:8]
	}
	return uuid
}

// displayName converts a snake_case / dash id back to a readable name.
// "wendy" → "Wendy", "jerin-peter" → "Jerin Peter".
func displayName(id string) string {
	if id == "" {
		return ""
	}
	cleaned := strings.ReplaceAll(id, "-", " ")
	cleaned = strings.ReplaceAll(cleaned, "_", " ")
	// Title-case each word
	words := strings.Fields(cleaned)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + strings.ToLower(w[1:])
		}
	}
	return strings.Join(words, " ")
}

func (c *Connector) dispatchResponse(resp map[string]any) {
	if resp == nil {
		c.writeStatus("result=network_error")
		c.speak("I couldn't undo that.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		// Python returns: {ok, uuid, name, identities, took_sec}
		uuid := strOr(resp, "uuid", "")
		name := strOr(resp, "name", "")
		c.writeStatus(fmt.Sprintf(
			"result=success uuid=%s name=%s",
			shortUUID(uuid), name,
		))
		if name != "" {
			c.speak(fmt.Sprintf("OK, I've forgotten %s. Let's try again.", displayName(name)))
		} else {
			// Anon UUID was forgotten — no name to say
			c.speak("OK, I've forgotten that one. Let's try again.")
		}
		c.log.Info("forget_last/http: ok",
			zap.String("uuid", uuid), zap.String("name", name))
		return
	}
	errStr := strOr(resp, "error", "unknown")
	switch errStr {
	case "no_recent_enrollment":
		c.writeStatus("result=no_recent_enrollment")
		c.speak("There's nothing recent for me to forget.")
	case "stale_enrollment":
		c.writeStatus("result=stale_enrollment")
		c.speak("Too much time has passed — I can't undo that anymore.")
	case "uuid_mismatch":
		// The action no longer sends a uuid, so this shouldn't fire
		// in normal use. Surface defensively in case of unexpected payload.
		detail := strOr(resp, "detail", "")
		c.writeStatus(fmt.Sprintf("result=uuid_mismatch detail=%s", detail))
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		c.speak("I can't undo that right now.")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		c.speak("Something went wrong undoing that.")
	}
}

func strOr(m map[string]any, k, d string) string {
	if s, ok := m[k].(string); ok {
		return s
	}
	return d
}