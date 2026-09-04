package tracer

import (
	"context"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/tracer/traceexport"
)

// startTraceExport starts broadcasting trace records on /metrics if
// enabled in cfg, using a context that can be cancelled on shutdown.
func (t *Tracer) startTraceExport(ctx context.Context, cfg *config.PrometheusExportConfig, log *zap.Logger) {
	if cfg == nil || !cfg.Enabled {
		return
	}

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
