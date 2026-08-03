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
	// persist before it's treated as a real barge-in
	defaultVADInterruptConfirmDelay = 150 * time.Millisecond

	// maxSanePendingAge bounds how long a VAD speech_end event stays eligible
	// to be paired with a transcript
	maxSanePendingAge = 20 * time.Second
)

// vadLatencyConfig optionally enables local VAD-vs-ASR latency measurement
// and/or VAD-driven TTS barge-in.
//
// If VADServiceURL is set, VAD inference runs on the remote GPU service at
// that URL (see docker/Dockerfile.vad in OM1-modules) instead of loading
// the Silero ONNX model locally and running it on CPU; VADModelPath and
// VADLibraryPath are then ignored.
type vadLatencyConfig struct {
	EnableVADLatency bool   `json:"enable_vad_latency"`
	VADModelPath     string `json:"vad_model_path"`
	VADLibraryPath   string `json:"vad_library_path"`
	VADOutputPath    string `json:"vad_output_path"`
	VADServiceURL    string `json:"vad_service_url"`

	VADInterruptConfirmMS int `json:"vad_interrupt_confirm_ms"`
}

// vadLatencyRecord is one JSONL line pairing a locally-detected end-of-speech
// moment with the ASR transcript that followed it.
type vadLatencyRecord struct {
	UtteranceEndedAt string  `json:"utterance_ended_at"`
	TranscriptAt     string  `json:"transcript_at"`
	LatencyMS        float64 `json:"latency_ms"`
	Provider         string  `json:"provider"`
	Transcript       string  `json:"transcript"`
}

// vadLatencyTracker runs a local Silero VAD alongside the ASR stream for
// latency measurement and TTS barge-in
type vadLatencyTracker struct {
	log        *zap.Logger
	model      vad.Backend
	segmenter  *vad.Segmenter
	outputPath string

	enableLatency         bool
	enableInterrupt       bool
	interruptConfirmDelay time.Duration

	mu         sync.Mutex
	pendingEnd time.Time
	lastStart  time.Time

	speechActive   bool
	candidateStart time.Time
	confirmed      bool
}

// newVADLatencyTracker builds a tracker from cfg, or returns nil if neither
// latency measurement nor TTS interrupt is requested
func newVADLatencyTracker(cfg vadLatencyConfig, enableTTSInterrupt bool, rate int, log *zap.Logger) *vadLatencyTracker {
	if !cfg.EnableVADLatency && !enableTTSInterrupt {
		return nil
	}

	model, modelFields, err := newVADBackend(cfg)
	if err != nil {
		fields := append(modelFields, zap.Error(err))
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

	log.Info("vad tracking enabled", append(modelFields,
		zap.String("output_path", outputPath),
		zap.Int("source_rate", rate),
		zap.Bool("latency_measurement", cfg.EnableVADLatency),
		zap.Bool("tts_interrupt", enableTTSInterrupt),
		zap.Duration("interrupt_confirm_delay", confirmDelay),
	)...)

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

// newVADBackend builds the Inferer backend selected by cfg: the remote GPU
// service if VADServiceURL is set, otherwise the local CPU ONNX model. The
// returned fields describe which backend was chosen and are meant to be
// logged on both the success and failure paths.
func newVADBackend(cfg vadLatencyConfig) (vad.Backend, []zap.Field, error) {
	if cfg.VADServiceURL != "" {
		fields := []zap.Field{
			zap.String("backend", "remote-gpu"),
			zap.String("service_url", cfg.VADServiceURL),
		}
		return vad.NewRemoteModel(cfg.VADServiceURL), fields, nil
	}

	modelPath := firstNonEmptyStr(cfg.VADModelPath, defaultVADModelPath)
	libPath := vad.ResolveLibraryPath(cfg.VADLibraryPath)
	fields := []zap.Field{
		zap.String("backend", "local-cpu"),
		zap.String("model_path", modelPath),
	}

	model, err := vad.NewModel(modelPath, libPath)
	if err != nil {
		return nil, fields, err
	}
	return model, fields, nil
}

// feedAudio runs the VAD over one PCM chunk: logs each speech boundary,
// records end-of-speech for later transcript pairing, and triggers TTS
// barge-in once speech clears the confirm delay
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
// out sub-confirm-delay blips
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
// unmatched VAD end-of-speech event, then records the latency
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

// close releases the underlying ONNX session
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
