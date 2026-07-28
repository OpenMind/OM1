package asr

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/vad"
)

const (
	defaultVADModelPath        = "models/silero_vad_v5.onnx"
	defaultVADLatencyOutputDir = "data/vad_asr_latency.jsonl"
)

// vadLatencyConfig is embedded in every ASR sensor config to optionally
// enable local Silero-VAD-vs-ASR latency measurement. It is opt-in: enabling
// it requires the onnxruntime shared library and the VAD model file to be
// present (see `make download-onnxruntime` and `make download-vad-model`).
type vadLatencyConfig struct {
	EnableVADLatency bool   `json:"enable_vad_latency"`
	VADModelPath     string `json:"vad_model_path"`
	VADLibraryPath   string `json:"vad_library_path"`
	VADOutputPath    string `json:"vad_output_path"`
}

// vadLatencyRecord is one JSONL line pairing a locally-detected end-of-speech
// moment with the ASR transcript that followed it.
type vadLatencyRecord struct {
	UtteranceEndedAt string  `json:"utterance_ended_at"` // VAD-detected end-of-speech, RFC3339Nano
	TranscriptAt     string  `json:"transcript_at"`      // ASR transcript acceptance time, RFC3339Nano
	LatencyMS        float64 `json:"latency_ms"`
	Provider         string  `json:"provider"`
	Transcript       string  `json:"transcript"`
}

// vadLatencyTracker runs a local Silero VAD alongside the ASR websocket
// stream and pairs each detected end-of-speech with the next accepted
// transcript, to measure how long the ASR vendor takes to return speech
// after the person actually stopped talking. All methods are nil-receiver
// safe so callers don't need to branch on whether the feature is enabled.
type vadLatencyTracker struct {
	log        *zap.Logger
	model      *vad.Model
	segmenter  *vad.Segmenter
	outputPath string

	mu      sync.Mutex
	pending []time.Time // FIFO of unmatched speech_end timestamps
}

// newVADLatencyTracker builds a tracker from cfg, or returns nil if disabled
// or if the VAD model/runtime can't be loaded. Load failures are logged and
// non-fatal: ASR keeps working without latency measurement.
func newVADLatencyTracker(cfg vadLatencyConfig, rate int, log *zap.Logger) *vadLatencyTracker {
	if !cfg.EnableVADLatency {
		return nil
	}

	modelPath := firstNonEmptyStr(cfg.VADModelPath, defaultVADModelPath)
	libPath := vad.ResolveLibraryPath(cfg.VADLibraryPath)

	model, err := vad.NewModel(modelPath, libPath)
	if err != nil {
		log.Warn("vad-asr latency disabled: failed to load Silero VAD model",
			zap.String("model_path", modelPath), zap.Error(err))
		return nil
	}

	outputPath := firstNonEmptyStr(cfg.VADOutputPath, defaultVADLatencyOutputDir)

	log.Info("vad-asr latency tracking enabled",
		zap.String("model_path", modelPath),
		zap.String("output_path", outputPath),
		zap.Int("source_rate", rate),
	)

	return &vadLatencyTracker{
		log:        log,
		model:      model,
		segmenter:  vad.NewSegmenter(model, rate, vad.SegmenterConfig{}),
		outputPath: outputPath,
	}
}

// feedAudio runs the VAD over one PCM chunk, recording any detected
// end-of-speech timestamp for later pairing with a transcript.
func (t *vadLatencyTracker) feedAudio(pcm []byte) {
	if t == nil {
		return
	}
	now := time.Now()

	t.mu.Lock()
	defer t.mu.Unlock()

	for _, ev := range t.segmenter.Feed(pcm, now) {
		if ev.Type != vad.EventSpeechEnd {
			continue
		}
		t.pending = append(t.pending, ev.At)
		t.log.Debug("vad detected end of speech", zap.Time("at", ev.At))
	}
}

// recordTranscript pairs an accepted transcript with the oldest unmatched
// VAD end-of-speech event and appends the resulting latency to the output
// file. Transcripts with no pending VAD event (e.g. arriving before the
// feature was enabled) are ignored.
func (t *vadLatencyTracker) recordTranscript(provider, text string) {
	if t == nil {
		return
	}
	now := time.Now()

	t.mu.Lock()
	if len(t.pending) == 0 {
		t.mu.Unlock()
		return
	}
	vadEnd := t.pending[0]
	t.pending = t.pending[1:]
	t.mu.Unlock()

	latency := now.Sub(vadEnd)
	rec := vadLatencyRecord{
		UtteranceEndedAt: vadEnd.Format(time.RFC3339Nano),
		TranscriptAt:     now.Format(time.RFC3339Nano),
		LatencyMS:        float64(latency.Microseconds()) / 1000.0,
		Provider:         provider,
		Transcript:       text,
	}

	if err := appendVADLatencyJSONL(t.outputPath, rec); err != nil {
		t.log.Warn("failed to write vad-asr latency record", zap.Error(err))
		return
	}

	t.log.Info("vad-asr latency",
		zap.Duration("latency", latency),
		zap.String("provider", provider),
		zap.String("transcript", truncateForLog(text, 60)),
	)
}

// close releases the underlying ONNX session, if one was loaded.
func (t *vadLatencyTracker) close() {
	if t == nil {
		return
	}
	if err := t.model.Close(); err != nil {
		t.log.Warn("failed to close vad model", zap.Error(err))
	}
}

func appendVADLatencyJSONL(path string, rec vadLatencyRecord) error {
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

func firstNonEmptyStr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

func truncateForLog(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
