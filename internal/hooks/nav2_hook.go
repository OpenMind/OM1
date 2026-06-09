package hooks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/providers/tts"
)

const (
	nav2LocalBaseURL = "http://localhost:5000"
	nav2CloudBaseURL = "https://api.openmind.com/api/core/simulation/orchestrator"
	nav2DefaultMap   = "map"
)

func init() {
	RegisterHook("nav2_hook", "start_nav2_hook", (*Runner).startNav2Hook)
	RegisterHook("nav2_hook", "stop_nav2_hook", (*Runner).stopNav2Hook)
}

func (r *Runner) startNav2Hook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := nav2BaseURL(cfg)
	apiKey := stringVal(cfg, "api_key")
	mapName := stringValDefault(cfg, "map_name", nav2DefaultMap)
	nav2URL := baseURL + "/start/nav2"

	payload, err := json.Marshal(map[string]any{"map_name": mapName})
	if err != nil {
		return err
	}

	r.log.Info("nav2: starting", zap.String("url", nav2URL), zap.String("map_name", mapName))
	if err := nav2Post(ctx, nav2URL, apiKey, payload, 5*time.Second); err != nil {
		r.log.Error("nav2: failed to start", zap.Error(err))
		return fmt.Errorf("failed to start Nav2: %w", err)
	}

	r.log.Info("nav2: started successfully")
	r.nav2TTS(cfg).AddText("Navigation system has started successfully.")
	return nil
}

func (r *Runner) stopNav2Hook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := nav2BaseURL(cfg)
	apiKey := stringVal(cfg, "api_key")
	nav2URL := baseURL + "/stop/nav2"

	r.log.Info("nav2: stopping", zap.String("url", nav2URL))
	if err := nav2Post(ctx, nav2URL, apiKey, nil, 5*time.Second); err != nil {
		r.log.Error("nav2: failed to stop", zap.Error(err))
		return fmt.Errorf("failed to stop Nav2: %w", err)
	}

	r.log.Info("nav2: stopped successfully")
	return nil
}

func nav2BaseURL(cfg map[string]any) string {
	if base := stringVal(cfg, "base_url"); base != "" {
		return base
	}
	if useSim, _ := cfg["use_sim"].(bool); useSim {
		return nav2CloudBaseURL
	}
	return nav2LocalBaseURL
}

func nav2Post(ctx context.Context, url, apiKey string, body []byte, timeout time.Duration) error {
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, reader)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", apiKey)

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		return fmt.Errorf("error calling Nav2 API: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s", nav2ErrorMessage(resp))
	}
	return nil
}

func nav2ErrorMessage(resp *http.Response) string {
	var body struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil || body.Message == "" {
		return "Unknown error"
	}
	return body.Message
}

func (r *Runner) nav2TTS(cfg map[string]any) *tts.ElevenLabsProvider {
	return tts.ElevenLabs(elevenLabsConfigFrom(cfg), r.log)
}
