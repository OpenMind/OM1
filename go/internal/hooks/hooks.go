package hooks

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/providers"
)

type HookType string

const (
	OnStartup HookType = "on_startup"
	OnEntry   HookType = "on_entry"
	OnExit    HookType = "on_exit"
	OnTimeout HookType = "on_timeout"
)

type Runner struct {
	hooks []config.HookSpec
	log   *zap.Logger
}

// New creates a new Runner with the given hooks and logger.
func New(hooks []config.HookSpec, log *zap.Logger) *Runner {
	return &Runner{hooks: hooks, log: log}
}

// Run executes all hooks matching the given HookType, passing vars for template formatting.
func (r *Runner) Run(ctx context.Context, hookType HookType, vars map[string]any) error {
	for _, h := range r.hooks {
		if HookType(h.HookType) != hookType {
			continue
		}
		if err := r.execute(ctx, h, vars); err != nil {
			r.log.Warn("hook failed",
				zap.String("type", h.HookType),
				zap.String("handler", h.HandlerType),
				zap.Error(err),
			)
		}
	}
	return nil
}

// execute runs a single hook based on its HandlerType.
// It supports "command" for shell commands, "message" and "action" for runtime-handled hooks.
func (r *Runner) execute(ctx context.Context, h config.HookSpec, vars map[string]any) error {
	switch h.HandlerType {
	case "command":
		cmd, _ := h.HandlerConfig["command"].(string)
		if cmd == "" {
			return fmt.Errorf("command hook missing 'command' field")
		}

		formatted := formatTemplate(cmd, vars)
		c := exec.CommandContext(ctx, "sh", "-c", formatted)

		var stdout, stderr bytes.Buffer
		c.Stdout = &stdout
		c.Stderr = &stderr
		if err := c.Run(); err != nil {
			r.log.Error("hook command failed",
				zap.String("stderr", strings.TrimSpace(stderr.String())),
				zap.Error(err),
			)
			return err
		}

		if out := strings.TrimSpace(stdout.String()); out != "" {
			r.log.Info("hook command output", zap.String("output", out))
		}

		return nil

	case "message":
		msg, _ := h.HandlerConfig["message"].(string)
		if msg == "" {
			return nil
		}
		formatted := formatTemplate(msg, vars)
		r.log.Info("lifecycle message", zap.String("message", formatted))

		cfg := providers.ElevenLabsConfig{
			APIKey:           stringVal(h.HandlerConfig, "api_key"),
			ElevenLabsAPIKey: stringVal(h.HandlerConfig, "elevenlabs_api_key"),
			VoiceID:          stringVal(h.HandlerConfig, "voice_id"),
			ModelID:          stringVal(h.HandlerConfig, "model_id"),
			OutputFormat:     stringVal(h.HandlerConfig, "output_format"),
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
		if rv, ok := h.HandlerConfig["rate"].(float64); ok && rv > 0 {
			cfg.Rate = int(rv)
		} else {
			cfg.Rate = providers.DefaultRate
		}

		providers.ElevenLabs(cfg, r.log).AddText(formatted)
		return nil

	case "action":
		r.log.Warn("lifecycle action: unimplemented handler type 'action', skipping")
		return nil

	default:
		return fmt.Errorf("unknown handler type %q", h.HandlerType)
	}
}

// formatTemplate replaces {var} in the template with corresponding values from vars.
func formatTemplate(s string, vars map[string]any) string {
	for k, v := range vars {
		s = strings.ReplaceAll(s, "{"+k+"}", fmt.Sprintf("%v", v))
	}
	return s
}

// stringVal safely extracts a string value from a map, returning empty string if not present or not a string.
func stringVal(m map[string]any, key string) string {
	v, _ := m[key].(string)
	return v
}
