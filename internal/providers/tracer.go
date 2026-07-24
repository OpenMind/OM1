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

// TraceRecord is one LLM interaction, written as a JSONL line and optionally
// handed to any in-process subscriber (see Tracer.Subscribe).
type TraceRecord struct {
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

	subscribers []chan<- TraceRecord // appended once at startup via Subscribe(), never removed
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

// Subscribe registers an in-process listener for every future TraceRecord and
// returns the channel it will be delivered on. Intended to be called once at
// startup (e.g. by internal/qualityscorer) before any tracing happens -- there
// is no Unsubscribe, since subscribers are expected to live for the process's
// lifetime. The channel is buffered; a subscriber that falls behind has
// records dropped for it rather than blocking Gauge() (see Gauge's comment).
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

	// Deliver to subscribers outside t.mu: SetGeneration() takes the same lock
	// and is called on every mode transition (runCortexLoop), and transitions
	// are processed strictly sequentially (handleModeTransitions). A subscriber
	// that's fallen behind must never be able to freeze every future mode
	// transition -- this is the same reasoning that previously moved this
	// tracer's (now-removed) Zenoh publish outside the lock, twice.
	for _, ch := range subs {
		select {
		case ch <- rec:
		default:
			logger.Get().Warn("tracer: subscriber channel full, dropping trace record")
		}
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
