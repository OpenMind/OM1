package asr

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/providers/tts"
	"github.com/openmind/om1/internal/vad"
)

const (
	defaultVADModelPath        = "models/silero_vad_v5.onnx"
	defaultVADLatencyOutputDir = "data/vad_asr_latency.jsonl"

	// defaultVADInterruptConfirmDelay is how long VAD-detected speech must
	// persist before it's treated as a real barge-in rather than a blip
	// (cough, click, brief noise). The Silero segmenter emits speech_start
	// on a single 32ms frame crossing the probability threshold with no
	// built-in debounce, so this confirm window is the only thing standing
	// between a blip and an unwanted TTS interrupt.
	defaultVADInterruptConfirmDelay = 150 * time.Millisecond

	// maxSanePendingAge bounds how long a VAD speech_end event stays eligible
	// to be paired with a transcript. Guards against a stale event (e.g. a
	// noise blip whose "transcript" never arrived because it was too short
	// to be accepted) silently pairing with a much later, unrelated
	// transcript and producing a nonsensical latency.
	maxSanePendingAge = 20 * time.Second
)

// vadLatencyConfig is embedded in every ASR sensor config to optionally
// enable local Silero-VAD-vs-ASR latency measurement and/or VAD-driven TTS
// barge-in. Both require the onnxruntime shared library and the VAD model
// file to be present (see `make download-onnxruntime` and `make
// download-vad-model`); if either is missing, VAD support is disabled for
// the sensor (logged, non-fatal) and it falls back to running without it.
type vadLatencyConfig struct {
	EnableVADLatency bool   `json:"enable_vad_latency"`
	VADModelPath     string `json:"vad_model_path"`
	VADLibraryPath   string `json:"vad_library_path"`
	VADOutputPath    string `json:"vad_output_path"`

	// VADInterruptConfirmMS is how long, in milliseconds, VAD-detected speech
	// must persist before triggering tts.RequestInterrupt. Only relevant when
	// the owning sensor has EnableTTSInterrupt set. Defaults to
	// defaultVADInterruptConfirmDelay if <=0.
	VADInterruptConfirmMS int `json:"vad_interrupt_confirm_ms"`
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
// stream. It serves two independent, optionally-combined purposes:
//   - latency measurement: pairs each detected end-of-speech with the next
//     accepted transcript, to measure how long the ASR vendor takes to
//     return speech after the person actually stopped talking (enableLatency).
//   - TTS barge-in: fires tts.RequestInterrupt as soon as sustained speech is
//     detected while TTS is playing, without waiting for a vendor transcript
//     (enableInterrupt).
//
// All methods are nil-receiver safe so callers don't need to branch on
// whether either feature is enabled.
type vadLatencyTracker struct {
	log        *zap.Logger
	model      *vad.Model
	segmenter  *vad.Segmenter
	outputPath string

	enableLatency         bool
	enableInterrupt       bool
	interruptConfirmDelay time.Duration

	mu         sync.Mutex
	pendingEnd time.Time // most recent unmatched speech_end, zero if none
	lastStart  time.Time // most recent speech_start, zero if none yet -- for logging utterance duration

	// interrupt confirm-delay state, guarded by mu; only used when enableInterrupt.
	speechActive   bool
	candidateStart time.Time // when the current speech_start was first seen, zero if none pending
	confirmed      bool      // whether candidateStart has already triggered an interrupt
}

// newVADLatencyTracker builds a tracker from cfg, or returns nil if neither
// latency measurement nor TTS interrupt is requested, or if the VAD
// model/runtime can't be loaded. Load failures are logged and non-fatal: ASR
// (and TTS interrupt, if requested) keeps working without VAD support.
func newVADLatencyTracker(cfg vadLatencyConfig, enableTTSInterrupt bool, rate int, log *zap.Logger) *vadLatencyTracker {
	if !cfg.EnableVADLatency && !enableTTSInterrupt {
		return nil
	}

	modelPath := firstNonEmptyStr(cfg.VADModelPath, defaultVADModelPath)
	libPath := vad.ResolveLibraryPath(cfg.VADLibraryPath)

	model, err := vad.NewModel(modelPath, libPath)
	if err != nil {
		fields := []zap.Field{zap.String("model_path", modelPath), zap.Error(err)}
		if enableTTSInterrupt {
			log.Error("vad-based tts interrupt disabled: failed to load Silero VAD model", fields...)
		} else {
			log.Warn("vad-asr latency disabled: failed to load Silero VAD model", fields...)
		}
		return nil
	}

	outputPath := firstNonEmptyStr(cfg.VADOutputPath, defaultVADLatencyOutputDir)

	confirmDelay := defaultVADInterruptConfirmDelay
	if cfg.VADInterruptConfirmMS > 0 {
		confirmDelay = time.Duration(cfg.VADInterruptConfirmMS) * time.Millisecond
	}

	log.Info("vad tracking enabled",
		zap.String("model_path", modelPath),
		zap.String("output_path", outputPath),
		zap.Int("source_rate", rate),
		zap.Bool("latency_measurement", cfg.EnableVADLatency),
		zap.Bool("tts_interrupt", enableTTSInterrupt),
		zap.Duration("interrupt_confirm_delay", confirmDelay),
	)

	return &vadLatencyTracker{
		log:                   log,
		model:                 model,
		segmenter:             vad.NewSegmenter(model, rate, vad.SegmenterConfig{}),
		outputPath:            outputPath,
		enableLatency:         cfg.EnableVADLatency,
		enableInterrupt:       enableTTSInterrupt,
		interruptConfirmDelay: confirmDelay,
	}
}

// feedAudio runs the VAD over one PCM chunk, logging every detected
// speech_start/speech_end boundary, recording end-of-speech timestamps for
// later pairing with a transcript (when latency measurement is enabled), and
// triggering a TTS barge-in once speech has been sustained past the confirm
// delay (when interrupt is enabled). A new speech_end always replaces any
// earlier unmatched one: VAD events and accepted transcripts aren't reliably
// 1:1 (short blips, ambient noise, or pauses can trigger a speech_end that
// never gets a transcript -- e.g. acceptASRTranscript rejects anything under
// 3 words), so only the most recent event is ever a plausible match for the
// *next* transcript.
func (t *vadLatencyTracker) feedAudio(pcm []byte) {
	if t == nil {
		return
	}
	now := time.Now()

	t.mu.Lock()
	defer t.mu.Unlock()

	for _, ev := range t.segmenter.Feed(pcm, now) {
		switch ev.Type {
		case vad.EventSpeechStart:
			t.lastStart = ev.At
			t.log.Info("vad: speech started", zap.Time("at", ev.At))
			if t.enableInterrupt {
				t.speechActive = true
				t.candidateStart = ev.At
				t.confirmed = false
			}
		case vad.EventSpeechEnd:
			if t.enableLatency {
				t.pendingEnd = ev.At
			}
			fields := []zap.Field{zap.Time("at", ev.At)}
			if !t.lastStart.IsZero() {
				fields = append(fields, zap.Duration("utterance_duration", ev.At.Sub(t.lastStart)))
			}
			t.log.Info("vad: speech ended", fields...)
			if t.enableInterrupt {
				t.speechActive = false
				t.candidateStart = time.Time{}
				t.confirmed = false
			}
		}
	}

	t.checkInterrupt(now)
}

// checkInterrupt fires tts.RequestInterrupt once speechActive has persisted
// past interruptConfirmDelay without an intervening speech_end, filtering
// out sub-confirm-delay blips. Split out from feedAudio so the confirm-delay
// decision can be unit tested without a live VAD segmenter. Callers must
// hold t.mu.
func (t *vadLatencyTracker) checkInterrupt(now time.Time) {
	if !t.enableInterrupt || !t.speechActive || t.confirmed {
		return
	}
	if now.Sub(t.candidateStart) < t.interruptConfirmDelay {
		return
	}

	t.confirmed = true
	if tts.Speaking.Load() {
		t.log.Info("vad: barge-in detected, interrupting TTS",
			zap.Duration("confirm_delay", now.Sub(t.candidateStart)))
		tts.RequestInterrupt()
	}
}

// recordTranscript pairs an accepted transcript with the most recent
// unmatched VAD end-of-speech event, appends the resulting latency to the
// output file, and feeds it into the om1_asr_latency_seconds Prometheus
// metric (labeled with language/apiVersion as reported by the caller).
// Transcripts with no pending VAD event (e.g. arriving before the feature
// was enabled) are ignored, as are pairings old enough to be obviously stale
// rather than a real ASR latency.
func (t *vadLatencyTracker) recordTranscript(provider, language, apiVersion, text string) {
	if t == nil {
		return
	}
	now := time.Now()

	t.mu.Lock()
	vadEnd := t.pendingEnd
	t.pendingEnd = time.Time{}
	t.mu.Unlock()

	if vadEnd.IsZero() {
		return
	}

	latency := now.Sub(vadEnd)
	if latency < 0 || latency > maxSanePendingAge {
		t.log.Warn("discarding implausible vad-asr latency pairing",
			zap.Duration("latency", latency),
			zap.String("provider", provider),
			zap.String("transcript", truncateForLog(text, 60)),
		)
		return
	}

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

	metrics.ASRLatency.WithLabelValues(provider, language, apiVersion).Observe(latency.Seconds())
	metrics.ASRLatencyLast.WithLabelValues(provider, language, apiVersion).Set(latency.Seconds())

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
