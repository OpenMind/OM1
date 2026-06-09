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
	slamLocalBaseURL = "http://localhost:5000"
	slamCloudBaseURL = "https://api.openmind.com/api/core/simulation/orchestrator"
	slamDefaultMap   = "map"
)

func init() {
	RegisterHook("slam_hook", "start_slam_hook", (*Runner).startSlamHook)
	RegisterHook("slam_hook", "stop_slam_hook", (*Runner).stopSlamHook)
}

func (r *Runner) startSlamHook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := slamBaseURL(cfg)
	apiKey := stringVal(cfg, "api_key")
	slamURL := baseURL + "/start/slam"

	r.log.Info("slam: starting", zap.String("url", slamURL))

	if err := slamPost(ctx, slamURL, apiKey, nil, 5*time.Second); err != nil {
		r.log.Error("slam: failed to start", zap.Error(err))
		return fmt.Errorf("failed to start SLAM: %w", err)
	}

	r.log.Info("slam: started successfully")
	return nil
}

func (r *Runner) stopSlamHook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := slamBaseURL(cfg)
	apiKey := stringVal(cfg, "api_key")
	mapName := stringValDefault(cfg, "map_name", slamDefaultMap)
	saveMapURL := baseURL + "/maps/save"
	stopSlamURL := baseURL + "/stop/slam"

	savePayload, err := json.Marshal(map[string]any{"map_name": mapName})
	if err != nil {
		return err
	}

	r.log.Info("slam: saving map", zap.String("url", saveMapURL), zap.String("map_name", mapName))
	if err := slamPost(ctx, saveMapURL, apiKey, savePayload, 10*time.Second); err != nil {
		r.log.Error("slam: failed to save map", zap.Error(err))
		return fmt.Errorf("failed to save SLAM map: %w", err)
	}
	r.log.Info("slam: map saved successfully")
	r.slamTTS(cfg).AddText("Map has been saved successfully.")

	r.log.Info("slam: stopping", zap.String("url", stopSlamURL))
	if err := slamPost(ctx, stopSlamURL, apiKey, nil, 10*time.Second); err != nil {
		r.log.Error("slam: failed to stop", zap.Error(err))
		return fmt.Errorf("failed to stop SLAM: %w", err)
	}

	r.log.Info("slam: stopped successfully")
	return nil
}

func slamBaseURL(cfg map[string]any) string {
	if base := stringVal(cfg, "base_url"); base != "" {
		return base
	}
	if useSim, _ := cfg["use_sim"].(bool); useSim {
		return slamCloudBaseURL
	}
	return slamLocalBaseURL
}

func slamPost(ctx context.Context, url, apiKey string, body []byte, timeout time.Duration) error {
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
		return fmt.Errorf("error calling SLAM API: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s", slamErrorMessage(resp))
	}
	return nil
}

func slamErrorMessage(resp *http.Response) string {
	var body struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil || body.Message == "" {
		return "Unknown error"
	}
	return body.Message
}

func (r *Runner) slamTTS(cfg map[string]any) *tts.ElevenLabsProvider {
	return tts.ElevenLabs(elevenLabsConfigFrom(cfg), r.log)
}
