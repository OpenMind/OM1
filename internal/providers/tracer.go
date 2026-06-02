package providers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/openmind/om1/internal/logger"
	"go.uber.org/zap"
)

// traceRecord is one JSONL line written by the Tracer.
type traceRecord struct {
	Timestamp  string           `json:"ts"`
	Generation int              `json:"generation"`
	LLMInput   string           `json:"llm_input"`
	LLMOutput  []map[string]any `json:"llm_output"`
}

// Tracer provides a simple mechanism to record LLM interactions.
type Tracer struct {
	mu          sync.Mutex
	outputDir   string
	enabled     bool
	currentDate string
	file        *os.File
	generation  int
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

// Gauge records an LLM interaction with the given input prompt and output.
func (t *Tracer) Gauge(llmInput string, llmOutput []map[string]any) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.enabled {
		return
	}

	if llmOutput == nil {
		llmOutput = []map[string]any{}
	}

	rec := traceRecord{
		Timestamp:  time.Now().UTC().Format(time.RFC3339Nano),
		Generation: t.generation,
		LLMInput:   llmInput,
		LLMOutput:  llmOutput,
	}

	line, err := json.Marshal(rec)
	if err != nil {
		logger.Get().Warn("tracer: failed to marshal record", zap.Error(err))
		return
	}

	if err := t.writeLocked(line); err != nil {
		logger.Get().Warn("tracer: failed to write record", zap.Error(err))
	}
}

// Stop closes the current file handle.
func (t *Tracer) Stop() {
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
		path := filepath.Join(t.outputDir, now+".jsonl")
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
