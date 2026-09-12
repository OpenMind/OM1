package tracer

import (
	"context"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/tracer/traceexport"
)

// startTraceExport starts broadcasting trace records on /metrics, using a
// context that can be cancelled on shutdown. It runs whenever tracing itself
// is enabled -- it's part of what the tracer does, not a separate opt-in.
func (t *Tracer) startTraceExport(ctx context.Context, log *zap.Logger) {
	teCtx, cancel := context.WithCancel(ctx)
	traceexport.Start(teCtx, t.Subscribe(), log)

	t.mu.Lock()
	t.traceExportCancel = cancel
	t.mu.Unlock()
}

// stopTraceExport cancels the trace exporter's context if it was started.
func (t *Tracer) stopTraceExport() {
	t.mu.Lock()
	cancel := t.traceExportCancel
	t.traceExportCancel = nil
	t.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}
