package hooks

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/memory"
	"github.com/openmind/om1/internal/providers/tts"
)

type HookType string

const (
	OnStartup HookType = "on_startup"
	OnEntry   HookType = "on_entry"
	OnExit    HookType = "on_exit"
	OnTimeout HookType = "on_timeout"
)

// Runner manages and executes lifecycle hooks based on their type and configuration.
type Runner struct {
	memory *memory.Manager
	hooks  []config.HookSpec
	log    *zap.Logger
}

// NewHooks creates a new Runner instance with the provided hooks, memory
// manager, and logger.
func NewHooks(hooks []config.HookSpec, memory *memory.Manager, log *zap.Logger) *Runner {
	return &Runner{hooks: hooks, memory: memory, log: log}
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

		cfg := elevenLabsConfigFrom(h.HandlerConfig)
		tts.ElevenLabs(cfg, r.log).AddText(formatted)
		return nil

	case "function":
		return r.executeFunction(ctx, h, vars)

	case "action":
		r.log.Warn("lifecycle action: unimplemented handler type 'action', skipping")
		return nil

	default:
		return fmt.Errorf("unknown handler type %q", h.HandlerType)
	}
}

// executeFunction looks up and executes a registered function hook based on the module and function names in the hook's HandlerConfig.
func (r *Runner) executeFunction(ctx context.Context, h config.HookSpec, vars map[string]any) error {
	module := stringVal(h.HandlerConfig, "module_name")
	fn := stringVal(h.HandlerConfig, "function")

	handler, ok := lookupHook(module, fn)
	if !ok {
		return fmt.Errorf("unknown function hook %s.%s", module, fn)
	}
	return handler(r, ctx, h.HandlerConfig, vars)
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

// elevenLabsConfigFrom builds an ElevenLabsConfig from a hook handler_config map,
// applying provider defaults for any field that is missing or empty.
func elevenLabsConfigFrom(m map[string]any) tts.ElevenLabsConfig {
	cfg := tts.ElevenLabsConfig{
		APIKey:           stringVal(m, "api_key"),
		ElevenLabsAPIKey: stringVal(m, "elevenlabs_api_key"),
		VoiceID:          stringVal(m, "voice_id"),
		ModelID:          stringVal(m, "model_id"),
		OutputFormat:     stringVal(m, "output_format"),
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
	if rv, ok := m["rate"].(float64); ok && rv > 0 {
		cfg.Rate = int(rv)
	} else {
		cfg.Rate = tts.DefaultRate
	}
	return cfg
}
