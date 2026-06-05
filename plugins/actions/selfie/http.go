package selfie

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
)

// SelfieInput is the LLM-facing schema for this action.
type SelfieInput struct {
	Action     string `json:"action" description:"The id to enroll, e.g. 'wendy'. Lowercase ASCII, dash/underscore allowed."`
	TimeoutSec int    `json:"timeout_sec" description:"Max seconds to wait for a face to appear (default 8)"`
	Force      bool   `json:"force" description:"Bypass cross-name reject (default false). Use for look-alike enrollment."`
}

func init() {
	actions.RegisterInterface(
		"selfie",
		"Enroll a person to the face gallery via the multi-frame /selfie endpoint. "+
			"The endpoint collects 1-4 quality-gated frames over a ~1.5s window, "+
			"selects the best target by engagement score, and either creates a new "+
			"identity or merges into an existing one. Outcomes surface to the LLM "+
			"via SelfieStatus with a brief TTS confirmation to the user.",
		SelfieInput{},
	)
	actions.Register("selfie", NewHTTPConnector)
}

// Config holds the connector's settings, parsed from JSON5 `config` block.
type Config struct {
	// TTS knobs (mirrored from speak/elevenlabs_tts.go)
	APIKey           string `json:"api_key"`
	ElevenLabsAPIKey string `json:"elevenlabs_api_key"`
	VoiceID          string `json:"voice_id"`
	ModelID          string `json:"model_id"`
	OutputFormat     string `json:"output_format"`
	Rate             int    `json:"rate"`

	// Face API knobs
	FaceHTTPBaseURL string  `json:"face_http_base_url"`
	FaceRecentSec   float64 `json:"face_recent_sec"`
	PollMs          int     `json:"poll_ms"`
	TimeoutSec      int     `json:"timeout_sec"`
	HTTPTimeoutSec  float64 `json:"http_timeout_sec"`
}

// Connector implements actions.Connector.
type Connector struct {
	log    *zap.Logger
	cfg    Config
	client *http.Client
	tts    *tts.ElevenLabsProvider

	mu             sync.Mutex
	lastEnrolledID string // breadcrumb for debugging; API owns the real TTL
	lastMatchName  string
}

// _DEDUP_SUFFIX_RE: strip trailing "_<digits>" e.g. "wendy_1" -> "wendy"
var dedupSuffixRE = regexp.MustCompile(`_\d+$`)

// NewHTTPConnector is registered as "selfie/http". The JSON5 config will
// instantiate it via {name: "selfie", connector: "http", config: {...}}.
func NewHTTPConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("selfie/http: api_key required")
	}
	if cfg.FaceHTTPBaseURL == "" {
		cfg.FaceHTTPBaseURL = "http://127.0.0.1:6793"
	}
	if cfg.FaceRecentSec == 0 {
		cfg.FaceRecentSec = 1.0
	}
	if cfg.PollMs == 0 {
		cfg.PollMs = 200
	}
	if cfg.TimeoutSec == 0 {
		cfg.TimeoutSec = 8
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

// Connect executes a single selfie enrollment attempt.
func (c *Connector) Connect(ctx context.Context, input actions.Input) (actions.Output, error) {
	args, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("selfie/http: unexpected input type %T", input)
	}

	name := strings.TrimSpace(stringArg(args, "action"))
	if name == "" {
		c.writeStatus("result=bad_id detail=empty")
		c.log.Error("selfie/http: empty id; LLM should produce something like 'wendy'")
		return nil, nil
	}

	timeoutSec := intArg(args, "timeout_sec", c.cfg.TimeoutSec)
	force := boolArg(args, "force", false)

	// Snapshot blur, disable for enrollment so the demo view shows the face.
	origBlur := c.getBlur()
	c.setBlur(false)
	defer c.setBlur(origBlur)

	// Pre-check: fast-fail if nobody is in frame.
	if !c.waitAnyFace(ctx, timeoutSec) {
		c.writeStatus("result=low_quality reason=no_one_present")
		c.speak("I don't see anyone in front of me yet.")
		return nil, nil
	}

	// Call /selfie — the API runs its multi-frame collection.
	body := map[string]any{"id": name, "force": force}
	resp := c.postJSON("/selfie", body)

	// Retry once on transient busy.
	if resp != nil {
		if errStr, _ := resp["error"].(string); errStr == "busy" {
			c.log.Info("selfie/http: busy on first call, retrying in 1s")
			select {
			case <-time.After(1 * time.Second):
			case <-ctx.Done():
				return nil, nil
			}
			resp = c.postJSON("/selfie", body)
		}
	}

	if resp == nil {
		c.dispatchNetworkError()
	} else {
		c.dispatchResponse(resp, name)
	}
	return nil, nil
}

// Tick is a no-op since this connector is event-driven (LLM calls Connect).
func (c *Connector) Tick(ctx context.Context) { <-ctx.Done() }

// Stop is a no-op since the shared ElevenLabs provider manages itself.
func (c *Connector) Stop() {}

// -------- HTTP helpers --------

func (c *Connector) postJSON(path string, body map[string]any) map[string]any {
	url := c.cfg.FaceHTTPBaseURL + path
	buf, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, bytes.NewReader(buf))
	if err != nil {
		c.log.Warn("selfie/http: build request failed", zap.String("url", url), zap.Error(err))
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.log.Warn("selfie/http: HTTP error", zap.String("url", url), zap.Error(err))
		return nil
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		c.log.Warn("selfie/http: decode failed", zap.String("url", url), zap.Error(err))
		return nil
	}
	return out
}

func (c *Connector) getBlur() bool {
	resp := c.postJSON("/config", map[string]any{"get": true})
	if resp == nil {
		return true
	}
	conf, _ := resp["config"].(map[string]any)
	v, ok := conf["blur"].(bool)
	if !ok {
		return true
	}
	return v
}

func (c *Connector) setBlur(on bool) {
	_ = c.postJSON("/config", map[string]any{"set": map[string]any{"blur": on}})
}

func (c *Connector) whoSnapshot() map[string]any {
	return c.postJSON("/who", map[string]any{"recent_sec": c.cfg.FaceRecentSec})
}

// waitAnyFace polls /who until ≥1 face appears, or timeout.
// Multi-person ambiguity is handled by the /selfie API itself.
func (c *Connector) waitAnyFace(ctx context.Context, timeoutSec int) bool {
	if timeoutSec <= 0 {
		timeoutSec = c.cfg.TimeoutSec
	}
	tries := (timeoutSec * 1000) / c.cfg.PollMs
	if tries < 1 {
		tries = 1
	}
	for i := 0; i < tries; i++ {
		resp := c.whoSnapshot()
		if resp != nil {
			now, _ := resp["now"].([]any)
			unknownNow, _ := resp["unknown_now"].(float64)
			if len(now)+int(unknownNow) >= 1 {
				c.log.Info("selfie/http: pre-check ok",
					zap.Int("known", len(now)),
					zap.Int("unknown", int(unknownNow)))
				return true
			}
		}
		select {
		case <-time.After(time.Duration(c.cfg.PollMs) * time.Millisecond):
		case <-ctx.Done():
			return false
		}
	}
	c.log.Info("selfie/http: pre-check no face within timeout",
		zap.Int("timeout_sec", timeoutSec))
	return false
}

// -------- Output helpers --------

// writeStatus surfaces a SelfieStatus line to the LLM on the next cortex
// tick. Uses the same IO channel as Voice (ASR) and FacePresence — it
// appears in subsequent prompts as `INPUT: SelfieStatus`.
//
// All three face-memory actions (selfie, correct_identity, forget_last)
// write to the single "SelfieStatus" key. The LLM disambiguates by
// reading the `result=...` prefix in the line.
func (c *Connector) writeStatus(line string) {
	providers.IO().AddInput("SelfieStatus", line, time.Time{})
	c.log.Info("selfie/http: status", zap.String("line", line))
}

func (c *Connector) speak(message string) {
	if message == "" {
		return
	}
	c.tts.AddText(message)
}

// _display_name: "wendy_1" → "Wendy", "jerin-peter" → "Jerin Peter"
func displayName(id string) string {
	cleaned := dedupSuffixRE.ReplaceAllString(id, "")
	cleaned = strings.ReplaceAll(cleaned, "-", " ")
	cleaned = strings.ReplaceAll(cleaned, "_", " ")
	return strings.Title(strings.ToLower(cleaned))
}

func (c *Connector) clearState() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastEnrolledID = ""
	c.lastMatchName = ""
}

// -------- Response dispatch --------

func (c *Connector) dispatchResponse(resp map[string]any, claimedID string) {
	// Success path
	if ok, _ := resp["ok"].(bool); ok {
		savedID := strOr(resp, "id", claimedID)
		merged := boolOr(resp, "merged", false)
		samples := intOr(resp, "samples_saved", 0)
		display := displayName(savedID)

		c.mu.Lock()
		c.lastEnrolledID = savedID
		c.lastMatchName = ""
		c.mu.Unlock()

		tag := "success"
		if merged {
			tag = "merged"
		}
		c.writeStatus(fmt.Sprintf("result=%s id=%s samples=%d merged=%t",
			tag, savedID, samples, merged))
		if merged {
			c.speak(fmt.Sprintf("Welcome back, %s!", display))
		} else {
			c.speak(fmt.Sprintf("Nice to meet you, %s! I'll remember you next time.", display))
		}
		c.log.Info("selfie/http: enroll ok",
			zap.String("tag", tag), zap.String("id", savedID), zap.Int("samples", samples))
		return
	}

	errStr := strOr(resp, "error", "unknown")

	switch errStr {
	case "ambiguous_subjects":
		n := intOr(resp, "n_engaged", 0)
		c.writeStatus(fmt.Sprintf("result=ambiguous engaged=%d", n))
		c.speak("I see a few people. Could you step closer so I can focus on you?")
		c.clearState()

	case "face_belongs_to":
		matched := strOr(resp, "name", "someone")
		sim := floatOr(resp, "sim", 0.0)
		display := displayName(matched)
		c.mu.Lock()
		c.lastMatchName = matched
		c.mu.Unlock()
		c.writeStatus(fmt.Sprintf("result=face_belongs_to claimed=%s matched=%s sim=%.3f",
			claimedID, matched, sim))
		c.speak(fmt.Sprintf("You look a lot like %s. Are you %s, or someone different?",
			display, display))

	case "no_valid_frames":
		c.writeStatus("result=low_quality")
		c.speak("I can't see your face clearly. Could you look at me directly?")
		c.clearState()

	case "insufficient_samples":
		got := intOr(resp, "got", 0)
		c.writeStatus(fmt.Sprintf("result=partial got=%d", got))
		c.speak("Hold still — almost got it.")
		c.clearState()

	case "busy":
		// Only reaches here AFTER the in-Connect retry
		c.writeStatus("result=busy retries=1")
		c.speak("One sec, finishing the last one.")
		c.clearState()

	case "bad_id":
		detail := strOr(resp, "detail", "")
		c.writeStatus(fmt.Sprintf("result=bad_id detail=%s", detail))
		// No TTS — LLM produced invalid id, should re-prompt user silently
		c.clearState()

	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		c.speak("I can't see right now — please try again in a moment.")
		c.clearState()

	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		c.speak("Something went wrong. Could you try again?")
		c.clearState()
	}
}

func (c *Connector) dispatchNetworkError() {
	c.writeStatus("result=network_error")
	c.speak("I lost connection for a moment.")
	c.clearState()
}

// -------- Tiny argument helpers --------

func stringArg(m map[string]any, k string) string  { s, _ := m[k].(string); return s }
func boolArg(m map[string]any, k string, d bool) bool {
	if v, ok := m[k].(bool); ok {
		return v
	}
	return d
}
func intArg(m map[string]any, k string, d int) int {
	switch v := m[k].(type) {
	case float64:
		return int(v)
	case int:
		return v
	}
	return d
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
func floatOr(m map[string]any, k string, d float64) float64 {
	if v, ok := m[k].(float64); ok {
		return v
	}
	return d
}