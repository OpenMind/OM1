// Package gallery_find_similar implements the similarity-search action.
//
// Calls /gallery/find_similar_current on the face API — the server-side
// endpoint that resolves the query UUID from the currently-largest visible
// face. The LLM does NOT pass any UUID. This is a deliberate ergonomic
// choice: 32-character hex strings are noise in LLM context and easy to
// hallucinate; "the person in front of me" is a stable, simpler reference.
//
// LLM WORKFLOW (read-only step)
// -----------------------------
// 1. LLM sees `anon_73d0a4 (met before, last seen ...)` in FacePresence.
// 2. Visitor says "do you remember me?"
// 3. LLM calls gallery_find_similar() with no arguments.
// 4. Server picks the largest visible face's UUID as the query.
// 5. Result lands in SimilarMatches as e.g.
//      "matches=2 best=sean(0.48) second=alice(0.32)"
//    The LLM reads this on its NEXT tick (user just spoke again) and
//    decides whether to ask "Are you Sean?".
//
// This action is READ-ONLY — does not modify the gallery. Follow-up
// confirmation goes through gallery_merge.
package gallery_find_similar

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

// GalleryFindSimilarInput is the LLM-facing schema.
//
// Notice: NO UUID field. The server resolves the query from the
// currently-visible face automatically.
type GalleryFindSimilarInput struct {
	TopK   int     `json:"top_k" description:"How many candidates to return (default 3). Larger top_k may include weaker matches."`
	MinSim float64 `json:"min_sim" description:"Minimum cosine sim threshold (default 0.30). Matches below this are filtered out."`
}

func init() {
	actions.RegisterInterface(
		"gallery_find_similar",
		"Look up which named gallery identities are visually similar to the "+
			"currently-visible face. Use BEFORE asking 'are you sean?' when "+
			"the visitor says 'do you remember me?' Server-side: uses the "+
			"largest face on screen as the query — you don't pass any UUID. "+
			"Read-only (does not modify the gallery).",
		GalleryFindSimilarInput{},
	)
	actions.Register("gallery_find_similar", NewHTTPConnector)
}

type Config struct {
	APIKey          string  `json:"api_key"`
	FaceHTTPBaseURL string  `json:"face_http_base_url"`
	HTTPTimeoutSec  float64 `json:"http_timeout_sec"`
	DefaultTopK     int     `json:"default_top_k"`
	DefaultMinSim   float64 `json:"default_min_sim"`
}

type Connector struct {
	log    *zap.Logger
	cfg    Config
	client *http.Client
}

func NewHTTPConnector(configMap map[string]any) (actions.Connector, error) {
	var cfg Config
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("gallery_find_similar/http: api_key required")
	}
	if cfg.FaceHTTPBaseURL == "" {
		cfg.FaceHTTPBaseURL = "http://127.0.0.1:6793"
	}
	if cfg.HTTPTimeoutSec == 0 {
		cfg.HTTPTimeoutSec = 5.0
	}
	if cfg.DefaultTopK == 0 {
		cfg.DefaultTopK = 3
	}
	if cfg.DefaultMinSim == 0 {
		cfg.DefaultMinSim = 0.30
	}

	return &Connector{
		log:    logger.Get(),
		cfg:    cfg,
		client: &http.Client{Timeout: time.Duration(cfg.HTTPTimeoutSec * float64(time.Second))},
	}, nil
}

func (c *Connector) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	args, _ := input.(map[string]any)
	if args == nil {
		args = map[string]any{}
	}

	topK := intOr(args, "top_k", c.cfg.DefaultTopK)
	if topK < 1 {
		topK = c.cfg.DefaultTopK
	}
	minSim := floatOr(args, "min_sim", c.cfg.DefaultMinSim)

	resp := c.postJSON("/gallery/find_similar_current", map[string]any{
		"top_k":   topK,
		"min_sim": minSim,
	})
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
		c.log.Warn("gallery_find_similar/http: build request failed", zap.Error(err))
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(req)
	if err != nil {
		c.log.Warn("gallery_find_similar/http: HTTP error", zap.Error(err))
		return nil
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		c.log.Warn("gallery_find_similar/http: decode failed", zap.Error(err))
		return nil
	}
	return out
}

// writeStatus writes a compact line to SimilarMatches that the LLM
// can read on the next tick. Format intentionally compact — no UUIDs,
// just name + sim, ranked.
func (c *Connector) writeStatus(line string) {
	providers.IO().AddInput("SimilarMatches", line, time.Time{})
	c.log.Info("gallery_find_similar/http: status", zap.String("line", line))
}

func (c *Connector) dispatchResponse(resp map[string]any) {
	if resp == nil {
		c.writeStatus("result=network_error")
		return
	}
	if ok, _ := resp["ok"].(bool); ok {
		rawMatches, _ := resp["matches"].([]any)
		if len(rawMatches) == 0 {
			c.writeStatus("result=no_match matches=0")
			return
		}
		// Compact summary line. Format:
		//   "result=success matches=2 best=sean(0.48) second=alice(0.32)"
		// Anon matches are surfaced as the anon_xxx short label.
		parts := []string{
			"result=success",
			fmt.Sprintf("matches=%d", len(rawMatches)),
		}
		for i, m := range rawMatches {
			match, ok := m.(map[string]any)
			if !ok {
				continue
			}
			name := strOr(match, "name", "")
			sim := floatOr(match, "sim", 0.0)
			label := name
			if label == "" {
				// Surface anon as short label
				uuid := strOr(match, "uuid", "")
				if len(uuid) >= 6 {
					label = "anon_" + uuid[:6]
				} else {
					label = "anon"
				}
			}
			slot := "extra"
			switch i {
			case 0:
				slot = "best"
			case 1:
				slot = "second"
			case 2:
				slot = "third"
			}
			parts = append(parts, fmt.Sprintf("%s=%s(%.2f)", slot, label, sim))
		}
		c.writeStatus(strings.Join(parts, " "))
		c.log.Info("gallery_find_similar/http: ok",
			zap.Int("matches", len(rawMatches)))
		return
	}
	errStr := strOr(resp, "error", "unknown")
	switch errStr {
	case "no_visible_face":
		c.writeStatus("result=no_visible_face")
	case "uuid_not_found":
		// Race: face_tracker has UUID but gallery doesn't (very rare)
		c.writeStatus("result=uuid_not_found")
	case "no_centroid":
		c.writeStatus("result=no_centroid")
	case "bad_top_k", "bad_min_sim":
		c.writeStatus("result=" + errStr)
	case "recognition_disabled":
		c.writeStatus("result=recognition_disabled")
	default:
		c.writeStatus(fmt.Sprintf("result=unknown error=%s", errStr))
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