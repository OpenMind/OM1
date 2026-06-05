// Package set_name_current implements "rename the person in front of me".
//
// Calls /set_name_current on the face API. The server resolves the
// CURRENTLY-LARGEST visible face to its UUID (exactly like
// merge_current / find_similar_current) and renames THAT UUID. The LLM
// passes only the new name — never a UUID, never the old name.
//
// WHY THIS EXISTS (vs correct_identity)
// -------------------------------------
// correct_identity resolves the target by NAME (from_id). That breaks
// when several identities share a display name ("two Wendys"): the
// face API returns result=ambiguous and refuses to guess. This action
// sidesteps the problem entirely by targeting the FACE on screen, not a
// name — so "call me yucheng now" works even with duplicate names.
//
// It is a pure rename (server-side /set_name), so it does NOT run the
// selfie enroll pipeline and never trips result=face_belongs_to.
//
// WHEN TO USE
// -----------
//   - A visible, already-known person wants a different name
//     ("actually, call me Yucheng").
//   - You need to rename ONE of several same-named people — point the
//     camera at them; the largest visible face is the target.
// For wrong-person captures use forget_last(); for look-alikes use
// selfie(force=true); to fold two UUIDs of the SAME person into one use
// gallery_merge.
package set_name_current

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
	"github.com/openmind/om1/internal/providers/tts"
)

// SetNameCurrentInput is the LLM-facing schema. Notice: NO uuid, NO from_id.
type SetNameCurrentInput struct {
	ToID string `json:"to_id" description:"New name for the person CURRENTLY in front of the camera (e.g. 'yucheng'). Lowercase ASCII, dashes for spaces. The server targets the largest visible face by face, so you do not pass any id or old name."`
}

func init() {
	actions.RegisterInterface(
		"set_name_current",
		"Rename the person CURRENTLY visible in front of the camera. Use when a "+
			"known/visible person wants to be called something different (e.g. "+
			"'call me yucheng now'), OR to rename one specific person when several "+
			"share a name. Server-side: targets the largest visible face by FACE "+
			"(not by name), so it works even with duplicate names. You pass only "+
			"to_id (the new name). Pure rename — does not re-enroll and never "+
			"reports face_belongs_to.",
		SetNameCurrentInput{},
	)
	actions.Register("set_name_current", NewHTTPConnector)
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
		return nil, fmt.Errorf("set_name_current/http: api_key required")
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
		return nil, fmt.Errorf("set_name_current/http: unexpected input type %T", input)
	}

	toID := normID(args, "to_id")
	if toID == "" {
		c.writeStatus("result=bad_id to=")
		c.log.Error("set_name_current/http: missing to_id")
		return nil, nil
	}

	resp := c.postJSON("/set_name_current", map[string]any{
		"name":         toID,
		"confirmed_by": "user_voice",
	})
	c.dispatchResponse(resp, toID)
	return nil, nil
}

func (c *Connector) Tick(ctx context.Context) { <-ctx.Done() }
func (c *Connector) Stop()                    {}

func (c *Connector) postJSON(path string, body map[string]any) map[string]any {
	url := c.cfg.FaceHTTPBaseURL + path
	buf, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, bytes.NewReader(buf))
	if err != nil {
		c.log.Warn("set_name_current/http: build request failed", zap.Error(err))
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.log.Warn("set_name_current/http: HTTP error", zap.Error(err))
		return nil
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		c.log.Warn("set_name_current/http: decode failed", zap.Error(err))
		return nil
	}
	return out
}

// writeStatus surfaces a SelfieStatus line to the LLM (same key the other
// face-memory actions use; the LLM disambiguates by the result= prefix).
func (c *Connector) writeStatus(line string) {
	providers.IO().AddInput("SelfieStatus", line, time.Time{})
	c.log.Info("set_name_current/http: status", zap.String("line", line))
}

func (c *Connector) speak(message string) {
	if message == "" {
		return
	}
	c.tts.AddText(message)
}

func (c *Connector) dispatchResponse(resp map[string]any, toID string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		c.speak("I had trouble updating that name.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		name := strOr(resp, "name", toID)
		prev := strOr(resp, "prev_name", "")
		uuid := strOr(resp, "uuid", "")
		c.writeStatus(fmt.Sprintf(
			"result=success name=%s prev_name=%s uuid=%s",
			name, prev, shortUUID(uuid),
		))
		c.speak(fmt.Sprintf("Got it — I'll call you %s from now on.", displayName(name)))
		c.log.Info("set_name_current/http: ok",
			zap.String("name", name), zap.String("prev", prev), zap.String("uuid", uuid))
		return
	}
	errStr := strOr(resp, "error", "unknown")
	switch errStr {
	case "no_visible_face":
		c.writeStatus("result=no_visible_face")
		c.speak("I can't see anyone in front of me right now.")
	case "bad_name":
		detail := strOr(resp, "detail", "")
		c.writeStatus(fmt.Sprintf("result=bad_id detail=%s", detail))
		// Silent — the LLM should re-prompt for a usable name.
	case "uuid_not_found":
		c.writeStatus("result=uuid_not_found")
		c.speak("I couldn't find that person anymore.")
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
		c.speak("I can't update names right now.")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		c.speak("Something went wrong updating that name.")
	}
}

func shortUUID(uuid string) string {
	if len(uuid) >= 8 {
		return uuid[:8]
	}
	return uuid
}

func displayName(id string) string {
	cleaned := dedupSuffixRE.ReplaceAllString(id, "")
	cleaned = strings.ReplaceAll(cleaned, "-", " ")
	cleaned = strings.ReplaceAll(cleaned, "_", " ")
	words := strings.Fields(cleaned)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + strings.ToLower(w[1:])
		}
	}
	return strings.Join(words, " ")
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
