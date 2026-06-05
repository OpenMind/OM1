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

	"golang.org/x/text/cases"
	"golang.org/x/text/language"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
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
	actions.Register("correct_identity", NewHTTPConnector)
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

	// 1. Resolve from_id (name) → uuid via /gallery/identities.
	uuid, count, err := c.resolveNameToUUID(fromID)
	if err != nil {
		c.writeStatus("result=network_error")
		c.speak("I had trouble updating that.")
		return nil, nil
	}
	switch count {
	case 0:
		c.writeStatus(fmt.Sprintf("result=name_not_found from=%s", fromID))
		c.speak(fmt.Sprintf("I don't have anyone named %s saved.", displayName(fromID)))
		return nil, nil
	case 1:
		// fall through to rename
	default:
		// Ambiguous — multiple UUIDs share this name. We refuse to guess.
		c.writeStatus(fmt.Sprintf("result=ambiguous from=%s count=%d", fromID, count))
		c.speak(fmt.Sprintf("I have more than one %s saved — I can't tell which one you meant.", displayName(fromID)))
		return nil, nil
	}

	// 2. Rename the resolved UUID to to_id via /set_name.
	resp := c.postJSON("/set_name", map[string]any{"uuid": uuid, "name": toID})
	c.dispatchSetNameResponse(resp, fromID, toID, uuid)
	return nil, nil
}

func (c *Connector) Tick(ctx context.Context) { <-ctx.Done() }
func (c *Connector) Stop()                    {}

// resolveNameToUUID looks up identities matching a display name (case-
// insensitive). Returns (uuid, count, err); only uuid is meaningful when
// count == 1.
func (c *Connector) resolveNameToUUID(name string) (string, int, error) {
	resp := c.postJSON("/gallery/identities", map[string]any{})
	if resp == nil {
		return "", 0, fmt.Errorf("network error")
	}
	rawList, _ := resp["identities"].([]any)
	target := strings.ToLower(strings.TrimSpace(name))
	matches := []string{}
	for _, item := range rawList {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		n, _ := m["name"].(string)
		if strings.ToLower(strings.TrimSpace(n)) == target {
			u, _ := m["uuid"].(string)
			if u != "" {
				matches = append(matches, u)
			}
		}
	}
	if len(matches) == 0 {
		return "", 0, nil
	}
	if len(matches) > 1 {
		return "", len(matches), nil
	}
	return matches[0], 1, nil
}

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
	return cases.Title(language.English).String(strings.ToLower(cleaned))
}

func (c *Connector) dispatchSetNameResponse(resp map[string]any, fromID, toID, uuid string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		c.speak("I had trouble updating that.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		// /set_name returned ok=true + name/sample_count fields.
		samples := intOr(resp, "sample_count", 0)
		c.writeStatus(fmt.Sprintf(
			"result=success from=%s to=%s uuid=%s samples=%d",
			fromID, toID, shortUUID(uuid), samples,
		))
		c.speak(fmt.Sprintf("Got it, I've updated your name to %s.", displayName(toID)))
		c.log.Info("correct_identity/http: ok",
			zap.String("from", fromID), zap.String("to", toID),
			zap.String("uuid", uuid), zap.Int("samples", samples))
		return
	}
	errStr := strOr(resp, "error", "unknown")
	switch errStr {
	case "uuid_not_found":
		// Race: someone deleted the UUID between our lookup and set_name.
		c.writeStatus(fmt.Sprintf("result=uuid_not_found uuid=%s", shortUUID(uuid)))
		c.speak("I couldn't find that identity anymore — maybe it was deleted.")
	case "bad_name":
		detail := strOr(resp, "detail", "")
		c.writeStatus(fmt.Sprintf("result=bad_id detail=%s", detail))
		// Silent — LLM should re-prompt
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		c.speak("I can't update names right now.")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		c.speak("Something went wrong updating that.")
	}
}

// shortUUID returns the first 8 chars of a UUID for compact logging.
func shortUUID(uuid string) string {
	if len(uuid) >= 8 {
		return uuid[:8]
	}
	return uuid
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
func intOr(m map[string]any, k string, d int) int {
	if v, ok := m[k].(float64); ok {
		return int(v)
	}
	return d
}
