package providers

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/openmind/om1/internal/logger"
	zenohsession "github.com/openmind/om1/internal/zenoh"
	"go.uber.org/zap"
)

// tracerEventTopic is the Zenoh topic each trace record is published to,
// live, alongside the JSONL file write.
const tracerEventTopic = "om/tracer/event"

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

	zenohOnce sync.Once
	publisher zenohsession.Publisher // nil if Zenoh is unavailable; publishing is then skipped
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
	t.enabled = true
	if err := os.MkdirAll(t.outputDir, 0o755); err != nil {
		logger.Get().Warn("tracer: failed to create output dir", zap.Error(err))
	}
	t.mu.Unlock()

	// Run async: Enable() is called from initializeMode, which onModeTransition
	// (runtime.go) runs on the single serialized mode-transition goroutine. If the
	// first tracer-enabled mode is entered via a transition rather than at startup,
	// a synchronous zenoh open here would still block every future mode transition
	// behind it. Firing it in a goroutine means Enable()/initializeMode always
	// return immediately; Gauge() just skips publishing until t.publisher is set.
	go t.zenohOnce.Do(t.initZenohPublisher)
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

	if t.publisher != nil {
		if err := t.publisher.Put(line); err != nil {
			logger.Get().Warn("tracer: failed to publish record", zap.Error(err))
		}
	}
}

// initZenohPublisher opens a Zenoh session and declares the tracer's publisher,
// caching the result (including failure, leaving t.publisher nil) for the
// Tracer's lifetime. Called once from Enable(), guarded by t.zenohOnce.
//
// Deliberately does NOT hold t.mu while opening the session: zenohsession.Open()
// can block for a while (client connect + discovery fallback, zenoh_backend.go),
// and t.mu is also taken by SetGeneration(), which runtime.runCortexLoop calls on
// every mode transition. Holding t.mu here previously meant a slow/unreachable
// Zenoh router stalled not just trace publishing but every future mode
// transition, since mode transitions are handled strictly sequentially. A
// missing Zenoh router degrades to file-only tracing rather than an error.
func (t *Tracer) initZenohPublisher() {
	sess, err := zenohsession.Open()
	if err != nil {
		logger.Get().Warn("tracer: zenoh unavailable, live trace publishing disabled", zap.Error(err))
		return
	}

	pub, err := sess.DeclarePublisher(tracerEventTopic)
	if err != nil {
		sess.Close()
		logger.Get().Warn("tracer: failed to declare publisher, live trace publishing disabled", zap.Error(err))
		return
	}

	t.mu.Lock()
	t.publisher = pub
	t.mu.Unlock()
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
