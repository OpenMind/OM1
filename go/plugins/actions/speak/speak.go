// Package speak registers the "speak" action plugin.
package speak

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
)

func init() {
	actions.RegisterInterface(
		"speak",
		"This action makes the robot speak a given text out loud.",
		SpeakInput{},
	)

	actions.Register("speak/passthrough", newPassthrough)
}

// ─── Input type ──────────────────────────────────────────────────────────────

// SpeakInput is the structured argument the LLM passes when invoking "speak".
type SpeakInput struct {
	Action string `json:"action" description:"The text to be spoken"`
}

// ─── Passthrough connector ────────────────────────────────────────────────────

type passthrough struct{ log *zap.Logger }

func newPassthrough(_ map[string]any) (actions.Connector, error) {
	logger, _ := zap.NewProduction()
	return &passthrough{log: logger}, nil
}

func (p *passthrough) Connect(_ context.Context, input actions.Input) (actions.Output, error) {
	arguments, ok := input.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("speak/passthrough: unexpected input type %T", input)
	}
	text, _ := arguments["action"].(string)
	p.log.Info("speak/passthrough", zap.String("text", text))
	return nil, nil
}

func (p *passthrough) Tick(ctx context.Context) {
	select {
	case <-ctx.Done():
	case <-time.After(60 * time.Second):
	}
}
func (p *passthrough) Stop()                  {}

