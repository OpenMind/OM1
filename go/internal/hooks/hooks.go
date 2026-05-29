package hooks

import (
	"context"
	"fmt"
	"os/exec"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
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

func New(hooks []config.HookSpec, log *zap.Logger) *Runner {
	return &Runner{hooks: hooks, log: log}
}

func (r *Runner) Run(ctx context.Context, hookType HookType) error {
	for _, h := range r.hooks {
		if HookType(h.HookType) != hookType {
			continue
		}
		if err := r.execute(ctx, h); err != nil {
			r.log.Warn("hook failed",
				zap.String("type", h.HookType),
				zap.String("handler", h.HandlerType),
				zap.Error(err),
			)
		}
	}
	return nil
}

func (r *Runner) execute(ctx context.Context, h config.HookSpec) error {
	switch h.HandlerType {
	case "command":
		cmd, _ := h.HandlerConfig["command"].(string)
		if cmd == "" {
			return fmt.Errorf("command hook missing 'command' field")
		}
		return exec.CommandContext(ctx, "sh", "-c", cmd).Run()

	case "message":
		msg, _ := h.HandlerConfig["message"].(string)
		r.log.Info("hook message", zap.String("message", msg))
		return nil

	case "action":
		r.log.Debug("action hook (handled by runtime)", zap.Any("config", h.HandlerConfig))
		return nil

	default:
		return fmt.Errorf("unknown handler type %q", h.HandlerType)
	}
}
