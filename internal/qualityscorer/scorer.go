// Package qualityscorer scores conversation quality (spoken language, input
// sentiment, prompt/response coherence) live as OM1 runs, and serves the
// results as Prometheus metrics for the Grafana "OM1 Quality Scores"
// dashboard. It replaces dataAnalysis's scripts/live_quality_scorer.py: where
// that ran as a separate process fed over a Zenoh topic, this subscribes
// directly to internal/providers.Tracer's in-process channel (see
// Tracer.Subscribe), so there is no serialization, no broker, and no second
// process to run.
package qualityscorer

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/providers"
)

const (
	// defaultModel and defaultBaseURL match plugins/llm/gemini.go's
	// defaults -- like every other LLM plugin in this repo, Gemini is
	// reached through OpenMind's own gateway (which does the Gemini-format
	// translation server-side), not Google's API directly, so requests stay
	// OpenAI-shaped and classify.go needs no changes.
	defaultModel   = "gemini-3.1-flash-lite"
	defaultBaseURL = "https://api.openmind.com/api/core/gemini"
	defaultLogPath = "data/live_quality_log.jsonl"

	// minCharsForLanguage matches live_quality_scorer.py's MIN_CHARS_FOR_LANGUAGE.
	minCharsForLanguage = 8
)

// Config configures the quality scorer. Populated from
// config.QualityScorerConfig by StartServer's caller.
type Config struct {
	Model   string
	BaseURL string
	APIKey  string
}

// logRecord is one JSONL line, matching live_quality_scorer.py's classification
// log record shape exactly.
type logRecord struct {
	ScoredAt            string `json:"scored_at"`
	TraceTS             string `json:"trace_ts"`
	Prompt              string `json:"prompt"`
	Response            string `json:"response"`
	Language            string `json:"language,omitempty"`
	InputClassification string `json:"input_classification"`
	Coherence           string `json:"coherence,omitempty"`
}

// Start subscribes to tracer's trace records and starts one goroutine
// scoring them as they arrive, until ctx is done. This starts no server of
// its own -- metrics are recorded via internal/metrics.RecordQualityTurn onto
// prometheus.DefaultRegisterer, the same registry internal/metrics' existing
// :9090 server already exposes at /metrics.
func Start(ctx context.Context, log *zap.Logger, tracer *providers.Tracer, cfg config.QualityScorerConfig) func() {
	resolved := Config{
		Model:   firstNonEmpty(cfg.Model, defaultModel),
		BaseURL: firstNonEmpty(cfg.BaseURL, defaultBaseURL),
		APIKey:  cfg.APIKey,
	}
	if resolved.APIKey == "" {
		log.Warn("qualityscorer: no api_key configured, quality scoring disabled")
		return func() {}
	}

	metrics.InitQualityLabels()
	initLanguageLabels()

	records := tracer.Subscribe()
	log.Info("qualityscorer: started, scoring live trace records",
		zap.String("model", resolved.Model), zap.String("base_url", resolved.BaseURL))

	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-ctx.Done():
				return
			case rec, ok := <-records:
				if !ok {
					return
				}
				scoreOne(ctx, log, resolved, rec)
			}
		}
	}()

	return func() {
		<-done
	}
}

func scoreOne(ctx context.Context, log *zap.Logger, cfg Config, rec providers.TraceRecord) {
	prompt := extractPrompt(rec.LLMInput)
	if prompt == "" {
		return
	}
	response, _ := extractResponse(rec.LLMOutput)

	var language string
	langText := response
	if langText == "" {
		langText = prompt
	}
	if len(langText) >= minCharsForLanguage {
		if code := detectLang(langText); code != "" {
			language = langName(code)
		}
	}

	label, err := classifyInput(ctx, cfg, prompt, response == "")
	if err != nil {
		log.Warn("qualityscorer: input classification failed, skipping turn", zap.Error(err))
		return
	}

	var coherence string
	if response != "" {
		coherence, err = classifyCoherence(ctx, cfg, prompt, response)
		if err != nil {
			log.Warn("qualityscorer: coherence classification failed, skipping coherence for this turn", zap.Error(err))
		}
	}

	scoredAt := time.Now().UTC()
	metrics.RecordQualityTurn(language, label, coherence)

	if err := appendJSONL(defaultLogPath, logRecord{
		ScoredAt:            scoredAt.Format(time.RFC3339Nano),
		TraceTS:             rec.Timestamp,
		Prompt:              prompt,
		Response:            response,
		Language:            language,
		InputClassification: label,
		Coherence:           coherence,
	}); err != nil {
		log.Warn("qualityscorer: failed to write classification log", zap.Error(err))
	}

	log.Info("qualityscorer: scored turn",
		zap.String("input_classification", label),
		zap.String("coherence", coherence),
		zap.String("language", language),
		zap.String("prompt", truncate(prompt, 60)),
	)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func appendJSONL(path string, rec logRecord) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	line, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open: %w", err)
	}
	defer func() { _ = f.Close() }()
	if _, err := f.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("write: %w", err)
	}
	return nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
