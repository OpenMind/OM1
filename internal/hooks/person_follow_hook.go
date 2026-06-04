package hooks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/httpclient"
	"github.com/openmind/om1/internal/providers/tts"
	"github.com/openmind/om1/internal/util"
)

const (
	followingBaseURL = "http://localhost:2000"
	visionBaseURL    = "http://localhost:2001"
)

func init() {
	RegisterHook("person_follow_hook", "start_person_follow_hook", (*Runner).startPersonFollowHook)
	RegisterHook("person_follow_hook", "switch_person_follow_hook", (*Runner).switchPersonFollowHook)
	RegisterHook("person_follow_hook", "stop_person_follow_hook", (*Runner).stopPersonFollowHook)
	RegisterHook("person_follow_hook", "set_mode_hook", (*Runner).setModeHook)
}

// startPersonFollowHook starts person-following mode by repeatedly issuing an
// enroll command and polling the vision system until a person is tracked.
func (r *Runner) startPersonFollowHook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := stringValDefault(cfg, "base_url", visionBaseURL)
	enrollTimeout := util.FloatFrom(cfg["enroll_timeout"], 3.0)
	maxRetries := int(util.FloatFrom(cfg["max_retries"], 5))

	tts := r.personFollowTTS()
	enrollURL := baseURL + "/enroll"
	statusURL := baseURL + "/status"

	for attempt := 0; attempt < maxRetries; attempt++ {
		r.log.Info("person follow: enrolling",
			zap.Int("attempt", attempt+1), zap.Int("max_retries", maxRetries))

		enrollCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		req, err := http.NewRequestWithContext(enrollCtx, http.MethodPost, enrollURL, nil)
		if err != nil {
			cancel()
			return err
		}

		resp, err := httpclient.Default().Do(req)
		if err != nil {
			cancel()
			r.log.Warn("person follow: enroll failed", zap.Error(err))
			continue
		}

		status := resp.StatusCode
		_ = resp.Body.Close()
		cancel()
		if status != http.StatusOK {
			continue
		}

		r.log.Info("person follow: enroll command sent")

		elapsed := 0.0
		for elapsed < enrollTimeout {
			if !util.Sleep(ctx, 500*time.Millisecond) {
				return ctx.Err()
			}
			elapsed += 0.5

			tracked, err := func() (bool, error) {
				statusCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
				defer cancel()

				req, err := http.NewRequestWithContext(statusCtx, http.MethodGet, statusURL, nil)
				if err != nil {
					return false, err
				}
				resp, err := httpclient.Default().Do(req)
				if err != nil {
					return false, err
				}
				defer func() { _ = resp.Body.Close() }()

				if resp.StatusCode != http.StatusOK {
					return false, fmt.Errorf("status endpoint returned %d", resp.StatusCode)
				}
				var body struct {
					IsTracked bool `json:"is_tracked"`
				}
				if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
					return false, err
				}
				return body.IsTracked, nil
			}()
			if err != nil {
				r.log.Warn("person follow: status poll failed", zap.Error(err))
				continue
			}
			if tracked {
				r.log.Info("person follow: tracking started")
				tts.AddText("I see you! I'll follow you now.")
				return nil
			}
		}

		r.log.Info("person follow: attempt not tracking, retrying", zap.Int("attempt", attempt+1))
	}

	r.log.Info("person follow: awaiting person detection")
	tts.AddText("Person following mode activated. Please stand in front of me.")
	return nil
}

// switchPersonFollowHook switches the person-following target to a new person.
func (r *Runner) switchPersonFollowHook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := stringValDefault(cfg, "base_url", visionBaseURL)
	tts := r.personFollowTTS()
	switchURL := baseURL + "/switch"

	r.log.Info("person follow: calling switch", zap.String("url", switchURL))

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, switchURL, nil)
	if err != nil {
		return err
	}

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		r.log.Error("person follow: switch error", zap.Error(err))
		tts.AddText("I couldn't connect to switch the person.")
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		r.log.Error("person follow: failed to switch", zap.Int("status", resp.StatusCode))
		tts.AddText("I couldn't switch to a new person.")
		return fmt.Errorf("switch failed with status %d", resp.StatusCode)
	}

	r.log.Info("person follow: switched successfully")
	tts.AddText("I'll follow a new person now.")
	return nil
}

// stopPersonFollowHook stops person-following mode by clearing the tracked person.
func (r *Runner) stopPersonFollowHook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := stringValDefault(cfg, "base_url", visionBaseURL)
	clearURL := baseURL + "/clear"

	r.log.Info("person follow: calling clear", zap.String("url", clearURL))

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, clearURL, nil)
	if err != nil {
		return err
	}

	resp, err := httpclient.Default().Do(req)
	if err != nil {
		r.log.Error("person follow: clear error", zap.Error(err))
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		r.log.Error("person follow: failed to clear", zap.Int("status", resp.StatusCode))
		return fmt.Errorf("clear failed with status %d", resp.StatusCode)
	}

	r.log.Info("person follow: cleared successfully")
	return nil
}

// setModeHook sets the person-follow mode (e.g. "greeting", "following").
func (r *Runner) setModeHook(ctx context.Context, cfg, _ map[string]any) error {
	baseURL := stringValDefault(cfg, "base_url", followingBaseURL)
	mode := stringVal(cfg, "mode")
	if mode == "" {
		return fmt.Errorf("set_mode_hook: missing required 'mode' field")
	}
	commandURL := baseURL + "/command"

	payload, err := json.Marshal(map[string]any{"cmd": "set_mode", "mode": mode})
	if err != nil {
		return err
	}

	r.log.Info("person follow: setting mode", zap.String("mode", mode), zap.String("url", commandURL))

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, commandURL, bytes.NewReader(payload))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	resp, err := httpclient.Default().Do(req)
	if err != nil {
		r.log.Error("person follow: set mode error", zap.Error(err))
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		r.log.Error("person follow: failed to set mode", zap.Int("status", resp.StatusCode))
		return fmt.Errorf("set mode failed with status %d", resp.StatusCode)
	}

	r.log.Info("person follow: mode set successfully", zap.String("mode", mode))
	return nil
}

// personFollowTTS returns the shared ElevenLabs provider used for spoken feedback.
func (r *Runner) personFollowTTS() *tts.ElevenLabsProvider {
	return tts.ElevenLabs(elevenLabsConfigFrom(nil), r.log)
}

// stringValDefault returns the string at key, or def if missing/empty.
func stringValDefault(m map[string]any, key, def string) string {
	if v := strings.TrimSpace(stringVal(m, key)); v != "" {
		return v
	}

	return def
}
