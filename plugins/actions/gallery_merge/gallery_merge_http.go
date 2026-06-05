// Package gallery_merge implements the explicit-merge action.
//
// Calls /gallery/merge_current on the face API — the server-side
// endpoint that resolves source and target UUIDs from natural-language
// inputs: source = the currently-visible face, target = the same-name
// UUID in the gallery closest to source by centroid sim. The LLM passes
// only target_name (the verified name) and confirmed_by (audit note).
//
// WHEN TO USE
// -----------
// AFTER confirming via dialog that two identities are the same person.
// Typical workflow:
//
//   1. LLM sees `anon_73d0a4 (met before, ...)` in FacePresence.
//   2. LLM calls gallery_find_similar() — server uses visible face.
//   3. SimilarMatches → best=sean(0.48).
//   4. LLM asks visitor: "Are you Sean? You look familiar."
//   5. Visitor confirms.
//   6. LLM calls gallery_merge(target_name="sean", confirmed_by="user_voice").
//
// The 2-step process is INTENTIONAL — face similarity alone is not
// trustworthy enough to merge automatically (different people can have
// similar faces under poor conditions). User-in-the-loop guards against
// false merges that would permanently pollute a named identity.
//
// REVERSIBILITY
// -------------
// Merge soft-deletes the source UUID (moves to _trash/). On regret,
// /gallery/restore (no payload) restores the most-recently soft-deleted
// UUID. The LLM doesn't currently expose this — operator-level only.
package gallery_merge

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

// GalleryMergeInput is the LLM-facing schema.
//
// Notice: NO source_uuid / target_uuid. The server resolves both from
// the visible scene + target_name.
type GalleryMergeInput struct {
	TargetName  string `json:"target_name" description:"Display name the visitor confirmed (e.g. 'sean'). Server will pick the matching UUID in the gallery."`
	ConfirmedBy string `json:"confirmed_by" description:"How the merge was confirmed (e.g. 'user_voice', 'operator'). Free-form audit string."`
}

func init() {
	actions.RegisterInterface(
		"gallery_merge",
		"Merge the currently-visible anonymous face into a named gallery "+
			"identity. EXPLICIT operation — only call after dialog confirmation "+
			"(typically following gallery_find_similar). Server-side: source = "+
			"the largest visible face, target = the same-name gallery UUID "+
			"closest to source by similarity. You only pass target_name + "+
			"confirmed_by. The source UUID is soft-deleted (restorable).",
		GalleryMergeInput{},
	)
	actions.Register("gallery_merge/http", NewHTTPConnector)
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
	// MinSim is the min centroid sim required when picking among multiple
	// same-name UUIDs. Below this, server reports ambiguous_target.
	MinSim float64 `json:"min_sim"`
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
		return nil, fmt.Errorf("gallery_merge/http: api_key required")
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
		return nil, fmt.Errorf("gallery_merge/http: unexpected input type %T", input)
	}

	targetName := strings.ToLower(strings.TrimSpace(strOr(args, "target_name", "")))
	confirmedBy := strings.TrimSpace(strOr(args, "confirmed_by", ""))

	if targetName == "" {
		c.writeStatus("result=missing_target_name")
		c.log.Warn("gallery_merge/http: missing target_name")
		return nil, nil
	}
	if confirmedBy == "" {
		// Default audit tag — LLM should ideally pass something more
		// specific (e.g. "user_voice"). Tolerate omission for robustness.
		confirmedBy = "llm_unattributed"
	}

	body := map[string]any{
		"target_name":  targetName,
		"confirmed_by": confirmedBy,
	}
	if c.cfg.MinSim > 0 {
		body["min_sim"] = c.cfg.MinSim
	}

	resp := c.postJSON("/gallery/merge_current", body)
	c.dispatchResponse(resp, targetName)
	return nil, nil
}

func (c *Connector) Tick(ctx context.Context) { <-ctx.Done() }
func (c *Connector) Stop()                    {}

func (c *Connector) postJSON(path string, body map[string]any) map[string]any {
	url := c.cfg.FaceHTTPBaseURL + path
	buf, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", url, bytes.NewReader(buf))
	if err != nil {
		c.log.Warn("gallery_merge/http: build request failed", zap.Error(err))
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.log.Warn("gallery_merge/http: HTTP error", zap.Error(err))
		return nil
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		c.log.Warn("gallery_merge/http: decode failed", zap.Error(err))
		return nil
	}
	return out
}

func (c *Connector) writeStatus(line string) {
	providers.IO().AddInput("SelfieStatus", line, time.Time{})
	c.log.Info("gallery_merge/http: status", zap.String("line", line))
}

func (c *Connector) speak(message string) {
	if message == "" {
		return
	}
	c.tts.AddText(message)
}

func (c *Connector) dispatchResponse(resp map[string]any, targetName string) {
	if resp == nil {
		c.writeStatus("result=network_error")
		c.speak("I had trouble combining those identities.")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		samples := intOr(resp, "samples_merged", 0)
		sim := floatOr(resp, "sim", 0.0)
		c.writeStatus(fmt.Sprintf(
			"result=success target_name=%s samples=%d sim=%.2f",
			targetName, samples, sim,
		))
		// Quiet success — LLM persona usually verbalizes via its own
		// reply ("Got it, Sean! Welcome back.").
		c.log.Info("gallery_merge/http: ok",
			zap.String("target_name", targetName),
			zap.Int("samples", samples), zap.Float64("sim", sim))
		return
	}
	errStr := strOr(resp, "error", "unknown")
	switch errStr {
	case "missing_target_name":
		c.writeStatus("result=missing_target_name")
	case "no_visible_face":
		c.writeStatus("result=no_visible_face")
		c.speak("I can't see anyone in front of me right now.")
	case "source_is_named":
		srcName := strOr(resp, "source_name", "")
		c.writeStatus(fmt.Sprintf(
			"result=source_is_named source_name=%s", srcName,
		))
		// LLM will see this and can recover ("you said you were sean
		// but I have you as alice — let me check")
	case "target_name_not_found":
		c.writeStatus(fmt.Sprintf(
			"result=target_name_not_found target_name=%s", targetName,
		))
		// Silent — LLM should reprompt or fall back to selfie() if the
		// visitor wants a fresh enrollment.
	case "ambiguous_target":
		c.writeStatus(fmt.Sprintf(
			"result=ambiguous_target target_name=%s", targetName,
		))
		c.speak("I have a few people with that name — could you help me figure out which?")
	case "no_centroid":
		c.writeStatus("result=no_centroid")
	case "same_uuid":
		c.writeStatus("result=same_uuid")
	case "uuid_not_found":
		role := strOr(resp, "role", "?")
		c.writeStatus(fmt.Sprintf("result=uuid_not_found role=%s", role))
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
		c.speak("Something went wrong combining those identities.")
	}
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
func floatOr(m map[string]any, k string, d float64) float64 {
	if v, ok := m[k].(float64); ok {
		return v
	}
	return d
}