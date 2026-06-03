// Package correct_identity implements the identity-rename action.
//
// Calls /gallery/move_samples on the face API to move all samples from
// one id to another. Used when the LLM/user wants to correct a recently-
// enrolled label (typo, mishearing) — NOT for wrong-person captures
// (use forget_last for that) or look-alikes (use selfie with force=true).
//
// The API enforces the 60s TTL on last_enrollment, so this connector
// can't accidentally rename an old identity even if the LLM asks it to.
package correct_identity

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

// CorrectIdentityInput is the LLM-facing schema for this action.
type CorrectIdentityInput struct {
	FromID string `json:"from_id" description:"The currently-wrong id, e.g. 'wendi'"`
	ToID   string `json:"to_id" description:"The corrected id, e.g. 'wendy'"`
}

func init() {
	actions.RegisterInterface(
		"correct_identity",
		"Rename a recently-enrolled identity to fix a label error. "+
			"Use ONLY when SelfieStatus showed result=success within ~60s AND the "+
			"user said it was a typo/mishearing of THE SAME PERSON. For look-alikes "+
			"use selfie(force=true); for wrong-person captures use forget_last().",
		CorrectIdentityInput{},
	)
	actions.Register("correct_identity/http", NewHTTPConnector)
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
	tts    *providers.ElevenLabsProvider
}

var dedupSuffixRE = regexp.MustCompile(`_\d+$`)

func NewHTTPConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("correct_identity/http: api_key required")
	}
	if cfg.FaceHTTPBaseURL == "" {
		cfg.FaceHTTPBaseURL = "http://127.0.0.1:6793"
	}
	if cfg.HTTPTimeoutSec == 0 {
		cfg.HTTPTimeoutSec = 5.0
	}
	if cfg.VoiceID == "" {
		cfg.VoiceID = providers.DefaultVoiceID
	}
	if cfg.ModelID == "" {
		cfg.ModelID = providers.DefaultModelID
	}
	if cfg.OutputFormat == "" {
		cfg.OutputFormat = providers.DefaultOutputFormat
	}
	if cfg.Rate == 0 {
		cfg.Rate = providers.DefaultRate
	}

	log := logger.Get()
	tts := providers.ElevenLabs(providers.ElevenLabsConfig{
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
		tts:    tts,
	}, nil
}

func (c *Connector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("correct_identity/http: unexpected input type %T", input)
	}

	fromID := normID(args, "from_id")
	toID := normID(args, "to_id")

	// Local validation — catch obvious LLM mistakes before HTTP
	if fromID == "" || toID == "" {
		c.writeStatus(fmt.Sprintf("result=bad_id from=%q to=%q", fromID, toID))
		c.log.Error("correct_identity/http: missing ids",
			zap.String("from_id", fromID), zap.String("to_id", toID))
		return nil, nil
	}
	if fromID == toID {
		c.writeStatus(fmt.Sprintf("result=same_id id=%s", fromID))
		c.log.Info("correct_identity/http: no-op (from_id == to_id)")
		return nil, nil
	}

	resp := c.postJSON("/gallery/move_samples",
		map[string]any{"from_id": fromID, "to_id": toID})
	c.dispatchResponse(resp, fromID, toID)
	return nil, nil
}

func (c *Connector) Tick(ctx context.Context) { <-ctx.Done() }
func (c *Connector) Stop()                    {}

func (c *Connector) postJSON(path string, body map[string]any) map[string]any {
	url := c.cfg.FaceHTTPBaseURL + path
	buf, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, bytes.NewReader(buf))
	if err != nil {
		c.log.Warn("correct_identity/http: build request failed", zap.Error(err))
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.log.Warn("correct_identity/http: HTTP error", zap.Error(err))
		return nil
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		c.log.Warn("correct_identity/http: decode failed", zap.Error(err))
		return nil
	}
	return out
}

// writeStatus surfaces a SelfieStatus line to the LLM. All three
// face-memory actions write to the same key — the LLM disambiguates by
// reading the `result=...` prefix.
func (c *Connector) writeStatus(line string) {
	providers.IO().AddInput("SelfieStatus", line, time.Time{})
	c.log.Info("correct_identity/http: status", zap.String("line", line))
}

func (c *Connector) speak(message string) {
	if message == "" {
		return
	}
	c.tts.AddText(message)
}

func displayName(id string) string {
	cleaned := dedupSuffixRE.ReplaceAllString(id, "")
	cleaned = strings.ReplaceAll(cleaned, "-", " ")
	cleaned = strings.ReplaceAll(cleaned, "_", " ")
	return strings.Title(strings.ToLower(cleaned))
}

func (c *Connector) dispatchResponse(resp map[string]any, fromID, toID string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		c.speak("I had trouble updating that.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		moved := intOr(resp, "moved", 0)
		fromRemoved := boolOr(resp, "from_removed", false)
		c.writeStatus(fmt.Sprintf("result=success from=%s to=%s moved=%d from_removed=%t",
			fromID, toID, moved, fromRemoved))
		c.speak(fmt.Sprintf("Got it, I've updated your name to %s.", displayName(toID)))
		c.log.Info("correct_identity/http: ok",
			zap.String("from", fromID), zap.String("to", toID), zap.Int("moved", moved))
		return
	}
	errStr := strOr(resp, "error", "unknown")
	switch errStr {
	case "no_recent_enrollment", "stale_enrollment":
		c.writeStatus("result=" + errStr)
		c.speak("I can't change that — too much time has passed since I remembered you.")
	case "bad_id":
		detail := strOr(resp, "detail", "")
		c.writeStatus(fmt.Sprintf("result=bad_id detail=%s", detail))
		// Silent — LLM should re-prompt
	case "same_id":
		c.writeStatus(fmt.Sprintf("result=same_id id=%s", fromID))
		// Silent
	case "no_safe_files":
		c.writeStatus("result=no_safe_files")
		c.speak("I couldn't find the right files to update.")
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		c.speak("I can't update names right now.")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		c.speak("Something went wrong updating that.")
	}
}

func normID(args map[string]any, k string) string {
	s, _ := args[k].(string)
	return strings.ToLower(strings.TrimSpace(s))
}
func strOr(m map[string]any, k, d string) string {
	if s, ok := m[k].(string); ok {
		return s
	}
	return d
}
func boolOr(m map[string]any, k string, d bool) bool {
	if v, ok := m[k].(bool); ok {
		return v
	}
	return d
}
func intOr(m map[string]any, k string, d int) int {
	if v, ok := m[k].(float64); ok {
		return int(v)
	}
	return d
}