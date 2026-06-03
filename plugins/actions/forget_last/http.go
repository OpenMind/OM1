// Package forget_last implements the undo-last-enrollment action.
//
// Calls /gallery/forget_last on the face API to delete the most recent
// enrollment's samples. Used when the WRONG PERSON was captured (someone
// walked in front of the camera, or the user wasn't actually facing it).
//
// NOT for typos (use correct_identity) or look-alikes (use selfie with
// force=true). The API enforces a 60s TTL so this can't accidentally
// forget an old enrollment.
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
)

// ForgetLastInput is the LLM-facing schema for this action.
type ForgetLastInput struct {
	ID string `json:"id" description:"Optional id of the enrollment to forget. If omitted, forgets the most recent one. The API will reject with id_mismatch if this doesn't match what was just enrolled."`
}

func init() {
	actions.RegisterInterface(
		"forget_last",
		"Undo the most recent /selfie enrollment by deleting its samples. "+
			"Use when SelfieStatus showed result=success within ~60s AND the user "+
			"indicates the WRONG PERSON was captured (e.g. 'someone walked in front', "+
			"'that wasn't me'). NOT for typos (use correct_identity) or look-alikes.",
		ForgetLastInput{},
	)
	actions.Register("forget_last/http", NewHTTPConnector)
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
	args, _ := input.(map[string]any)
	idCheck, _ := args["id"].(string)
	idCheck = strings.ToLower(strings.TrimSpace(idCheck))

	body := map[string]any{}
	if idCheck != "" {
		body["id"] = idCheck
	}

	resp := c.postJSON("/gallery/forget_last", body)
	c.dispatchResponse(resp, idCheck)
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

// writeStatus surfaces a SelfieStatus line to the LLM. All three
// face-memory actions write to the same key — the LLM disambiguates by
// reading the `result=...` prefix.
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

func (c *Connector) dispatchResponse(resp map[string]any, requestedID string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		c.speak("I couldn't undo that.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		forgotten := strOr(resp, "id", "")
		deleted := intOr(resp, "files_deleted", 0)
		identityRemoved := boolOr(resp, "identity_removed", false)
		c.writeStatus(fmt.Sprintf("result=success id=%s files_deleted=%d identity_removed=%t",
			forgotten, deleted, identityRemoved))
		c.speak("OK, I've forgotten that one. Let's try again.")
		c.log.Info("forget_last/http: ok",
			zap.String("id", forgotten), zap.Int("deleted", deleted),
			zap.Bool("identity_removed", identityRemoved))
		return
	}
	errStr := strOr(resp, "error", "unknown")
	switch errStr {
	case "no_recent_enrollment", "stale_enrollment":
		c.writeStatus("result=" + errStr)
		c.speak("There's nothing recent for me to forget.")
	case "id_mismatch":
		detail := strOr(resp, "detail", "")
		c.writeStatus(fmt.Sprintf("result=id_mismatch requested=%s detail=%s", requestedID, detail))
		c.speak("That name doesn't match what I just remembered.")
	case "no_safe_files":
		c.writeStatus("result=no_safe_files")
		c.speak("I couldn't find the right files to remove.")
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