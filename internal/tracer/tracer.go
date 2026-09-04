package tracer

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/tracer/tracetype"
	"go.uber.org/zap"
)

// TraceRecord is one LLM interaction, written as a JSONL line and handed to any subscriber.
type TraceRecord = tracetype.TraceRecord

// Tracer provides a simple mechanism to record LLM interactions.
type Tracer struct {
	mu          sync.Mutex
	outputDir   string
	enabled     bool
	currentDate string
	file        *os.File
	generation  int

	subscribers []chan<- TraceRecord

	qualityScoreCancel context.CancelFunc
	traceExportCancel  context.CancelFunc
}

var (
	tracerOnce     sync.Once
	tracerInstance *Tracer
)

// TracerProvider returns the singleton Tracer instance.
func TracerProvider() *Tracer {
	tracerOnce.Do(func() {
		tracerInstance = &Tracer{outputDir: "traces"}
	})
	return tracerInstance
}

// Start starts the tracer and its quality scorer if enabled in cfg, using a context that can be cancelled on shutdown.
func (t *Tracer) Start(ctx context.Context, cfg *config.TracerConfig, systemAPIKey string, log *zap.Logger) {
	if cfg == nil {
		return
	}

	if !cfg.Enabled {
		if cfg.QualityScorer != nil && cfg.QualityScorer.Enabled {
			log.Warn("tracer: quality_scorer.enabled is true but use_tracer.enabled is false -- quality scorer will not start")
		}
		if cfg.PrometheusExport != nil && cfg.PrometheusExport.Enabled {
			log.Warn("tracer: prometheus_export.enabled is true but use_tracer.enabled is false -- Prometheus trace export will not start")
		}
		return
	}
	t.Enable()

	t.startQualityScore(ctx, cfg.QualityScorer, systemAPIKey, log)
	t.startTraceExport(ctx, cfg.PrometheusExport, log)
}

// Enable turns tracing on and ensures the output directory exists.
func (t *Tracer) Enable() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.enabled = true
	if err := os.MkdirAll(t.outputDir, 0o755); err != nil {
		logger.Get().Warn("tracer: failed to create output dir", zap.Error(err))
	}
}

// Disable turns tracing off and closes any open file handle.
func (t *Tracer) Disable() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.enabled = false
	t.stopLocked()
}

// Enabled reports whether tracing is currently enabled.
func (t *Tracer) Enabled() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.enabled
}

// SetGeneration sets the current generation number recorded with each trace.
func (t *Tracer) SetGeneration(generation int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.generation = generation
}

// Subscribe returns a channel receiving every future TraceRecord; buffered and non-blocking, no Unsubscribe.
func (t *Tracer) Subscribe() <-chan TraceRecord {
	ch := make(chan TraceRecord, 32)
	t.mu.Lock()
	t.subscribers = append(t.subscribers, ch)
	t.mu.Unlock()
	return ch
}

// Gauge records an LLM interaction with the given input prompt and output.
func (t *Tracer) Gauge(llmInput string, llmOutput []map[string]any) {
	t.mu.Lock()

	if !t.enabled {
		t.mu.Unlock()
		return
	}

	if llmOutput == nil {
		llmOutput = []map[string]any{}
	}

	rec := TraceRecord{
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		Generation: t.generation,
		LLMInput:   llmInput,
		LLMOutput:  llmOutput,
	}

	line, err := json.Marshal(rec)
	if err != nil {
		t.mu.Unlock()
		logger.Get().Warn("tracer: failed to marshal record", zap.Error(err))
		return
	}

	if err := t.writeLocked(line); err != nil {
		logger.Get().Warn("tracer: failed to write record", zap.Error(err))
	}

	subs := t.subscribers
	t.mu.Unlock()

	for _, ch := range subs {
		select {
		case ch <- rec:
		default:
			logger.Get().Warn("tracer: subscriber channel full, dropping trace record")
		}
	}
}

// Stop stops the tracer, its quality scorer, and its trace exporter, closing any open file handle.
func (t *Tracer) Stop() {
	t.stopQualityScore()
	t.stopTraceExport()

	t.mu.Lock()
	defer t.mu.Unlock()
	t.stopLocked()
}

// stopLocked closes the current file handle without acquiring the mutex.
func (t *Tracer) stopLocked() {
	if t.file != nil {
		_ = t.file.Close()
		t.file = nil
		t.currentDate = ""
	}
}

// writeLocked writes a line to the current trace file, rotating daily.
func (t *Tracer) writeLocked(line []byte) error {
	now := time.Now().UTC().Format("2006-01-02")

	if now != t.currentDate {
		t.stopLocked()
		path := filepath.Join(t.outputDir, "tracer_"+now+".jsonl")
		f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err != nil {
			return err
		}
		t.file = f
		t.currentDate = now
	}

	if _, err := t.file.Write(append(line, '\n')); err != nil {
		return err
	}
	return t.file.Sync()
}
