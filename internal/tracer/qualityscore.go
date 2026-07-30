package tracer

import (
	"context"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/tracer/qualityscore"
)

// Start the quality scorer if enabled in cfg, using a context that can be cancelled on shutdown.
func (t *Tracer) startQualityScore(ctx context.Context, cfg *config.QualityScorerConfig, systemAPIKey string, log *zap.Logger) {
	if cfg == nil || !cfg.Enabled {
		return
	}

	qsCfg := *cfg
	if qsCfg.APIKey == "" {
		qsCfg.APIKey = systemAPIKey
	}

	qsCtx, cancel := context.WithCancel(ctx)
	qualityscore.Start(qsCtx, qsCfg, t.Subscribe(), log)

	t.mu.Lock()
	t.qualityScoreCancel = cancel
	t.mu.Unlock()
}

// stopQualityScore cancels the quality scorer's context if it was started.
func (t *Tracer) stopQualityScore() {
	t.mu.Lock()
	cancel := t.qualityScoreCancel
	t.qualityScoreCancel = nil
	t.mu.Unlock()

	if cancel != nil {
		cancel()
	}
}
