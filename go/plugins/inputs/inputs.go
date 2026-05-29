// Package inputs registers all built-in sensor plugins.
// Each concrete sensor lives in its own sub-package; this file provides
// a minimal "passthrough" sensor for testing and wires the registry.
package inputs

import (
	"context"
	"time"

	"github.com/openmind/om1/internal/inputs"
)

func init() {
	inputs.Register("PassthroughInput", newPassthrough)
}

// ─── PassthroughInput: emits a static string every interval ──────────────────

type passthroughInput struct {
	text     string
	interval time.Duration
	latest   string
}

func newPassthrough(configMap map[string]any) (inputs.Sensor, error) {
	text, _ := configMap["text"].(string)
	if text == "" {
		text = "no input"
	}
	intervalSeconds, _ := configMap["interval_seconds"].(float64)
	if intervalSeconds <= 0 {
		intervalSeconds = 1
	}
	return &passthroughInput{
		text:     text,
		interval: time.Duration(intervalSeconds * float64(time.Second)),
	}, nil
}

func (p *passthroughInput) RawToText(_ context.Context, rawValue any) (*inputs.Message, error) {
	text, _ := rawValue.(string)
	return &inputs.Message{Text: text, RawText: rawValue}, nil
}

// Listen spawns a goroutine that ticks every interval and sends the static
// text on the returned channel.  The goroutine exits when ctx is cancelled,
// closing the channel to signal EOF to the orchestrator.
func (p *passthroughInput) Listen(ctx context.Context) (<-chan any, error) {
	channel := make(chan any)
	go func() {
		defer close(channel)
		ticker := time.NewTicker(p.interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				channel <- p.text
			case <-ctx.Done():
				return
			}
		}
	}()
	return channel, nil
}

func (p *passthroughInput) LatestBuffer() string { return p.latest }

// Stop is a no-op for passthrough: lifecycle is managed entirely by the ctx
// passed to Listen.  Real sensors override this to close connections, etc.
func (p *passthroughInput) Stop() {}
