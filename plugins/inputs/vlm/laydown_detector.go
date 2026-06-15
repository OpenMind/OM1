package vlm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
	video "github.com/openmind/om1/internal/providers/vlm"
)

// PersonLaydownDetector is an RTSP+VLM input for spotting a downed person: it
// latches the ALERT with debounce and reports the verdict on every cortex tick.
func init() {
	inputs.Register("PersonLaydownDetector", NewPersonLaydownDetector)
}

const defaultClearStreak = 3

// laydownBufferSize is the frame-channel depth, sized so frames produced during a
// slow VLM call accumulate instead of being dropped.
const laydownBufferSize = 16

// defaultMaxBatch caps frames sent in one multi-image request, bounding latency and cost.
const defaultMaxBatch = 6

var laydownDefaults = providerDefaults{
	baseURL: geminiDefaults.baseURL,
	model:   geminiDefaults.model,
	prompt: "You are the safety vision system of a patrol robot. You are shown one or more " +
		"sequential video frames from the robot's camera. Look carefully at EVERY frame. " +
		"Decide whether a PERSON is lying on the ground or floor in ANY of the frames — collapsed, " +
		"fallen, slumped, or otherwise not standing or sitting upright in a normal way. Respond in " +
		"ONE line. If such a person appears in any frame, respond exactly in this form: 'ALERT: a " +
		"person is lying on the ground. <brief description of their position and any visible signs of " +
		"a medical condition>.' If no person is lying on the ground in any frame, respond exactly: " +
		"'No person lying on the ground.' Do not explain your reasoning.",
	maxTokens: 1024,
}

type laydownDetector struct {
	name        string
	log         *zap.Logger
	describer   *video.Describer
	source      frameSource
	clearStreak int
	minHold     time.Duration
	maxBatch    int
	debugDir    string

	mu         sync.Mutex
	latest     inputs.Message
	hasLatest  bool
	alerted    bool
	alertedAt  time.Time
	clearCount int
	lastAlert  string
	stopped    bool
	cancel     context.CancelFunc
}

// NewPersonLaydownDetector builds the latching RTSP laydown detector.
func NewPersonLaydownDetector(configMap map[string]any) (inputs.Sensor, error) {
	cfg, err := parseConfig(configMap, laydownDefaults)
	if err != nil {
		return nil, err
	}
	if cfg.RTSPURL == "" {
		cfg.RTSPURL = defaultRTSPURL
	}

	var extra struct {
		ClearStreak      int     `json:"clear_streak"`
		MinHoldSeconds   float64 `json:"min_hold_seconds"`
		MaxFramesPerCall int     `json:"max_frames_per_call"`
		ImageLogDir      string  `json:"image_log_dir"`
	}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &extra)
	}
	clearStreak := extra.ClearStreak
	if clearStreak <= 0 {
		clearStreak = defaultClearStreak
	}
	minHold := time.Duration(extra.MinHoldSeconds * float64(time.Second))
	maxBatch := extra.MaxFramesPerCall
	if maxBatch <= 0 {
		maxBatch = defaultMaxBatch
	}

	log := logger.Get().Named("PersonLaydownDetector")

	debugDir := extra.ImageLogDir
	if debugDir != "" {
		if err := os.MkdirAll(debugDir, 0o755); err != nil {
			log.Warn("image_log_dir unusable, disabling image dump",
				zap.String("dir", debugDir), zap.Error(err))
			debugDir = ""
		}
	}

	log.Info("initializing",
		zap.String("rtsp_url", cfg.RTSPURL),
		zap.String("model", cfg.Model),
		zap.Int("fps", cfg.FPS),
		zap.Int("clear_streak", clearStreak),
		zap.Duration("min_hold", minHold),
		zap.Int("max_frames_per_call", maxBatch),
		zap.String("image_log_dir", debugDir),
	)

	source := video.NewVideoRTSPStream(video.VideoRTSPStreamConfig{
		RTSPURL:     cfg.RTSPURL,
		FPS:         cfg.FPS,
		Width:       cfg.Width,
		Height:      cfg.Height,
		JPEGQuality: cfg.JPEGQuality,
		BufferSize: laydownBufferSize,
	})

	return &laydownDetector{
		name:        "PersonLaydownDetector",
		log:         log,
		source:      source,
		clearStreak: clearStreak,
		minHold:     minHold,
		maxBatch:    maxBatch,
		debugDir:    debugDir,
		describer: video.NewDescriber(video.Describer{
			Name:    "PersonLaydownDetector",
			APIKey:  cfg.APIKey,
			BaseURL: cfg.BaseURL,
			Model:   cfg.Model,
			Prompt:  cfg.Prompt,
			Detail:    "high",
			MaxTokens: cfg.MaxTokens,
			Log:       log,
		}),
	}, nil
}

// Listen batches the queued frames per VLM call and emits the latched verdict.
func (s *laydownDetector) Listen(ctx context.Context) (<-chan any, error) {
	ctx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.cancel = cancel
	s.mu.Unlock()

	frames := s.source.Start(ctx)
	out := make(chan any)

	go func() {
		defer close(out)
		defer s.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case frame, ok := <-frames:
				if !ok {
					return
				}

				batch := drainAllFrames(frames, frame, s.maxBatch)

				imgs := make([]string, 0, len(batch))
				for _, f := range batch {
					imgs = append(imgs, base64.StdEncoding.EncodeToString(f.JPEG))
				}
				newest := batch[len(batch)-1]
				oldest := batch[0]
				providers.LatestFrame().Set(newest.JPEG, newest.Timestamp)

				start := time.Now()
				text, err := s.describer.DescribeImages(ctx, imgs)
				callLatency := time.Since(start)
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					s.log.Warn("vision request failed", zap.Error(err))
					continue
				}

				s.log.Info("vlm latency",
					zap.Int("frames", len(batch)),
					zap.Int64("call_ms", callLatency.Milliseconds()),
					zap.Int64("newest_age_ms", time.Since(newest.Timestamp).Milliseconds()),
					zap.Int64("window_ms", newest.Timestamp.Sub(oldest.Timestamp).Milliseconds()),
					zap.String("verdict", text),
				)

				if s.debugDir != "" {
					s.dumpDebug(batch, text,
						callLatency.Milliseconds(),
						time.Since(newest.Timestamp).Milliseconds(),
						newest.Timestamp.Sub(oldest.Timestamp).Milliseconds())
				}

				if text == "" {
					continue
				}

				reading := s.classify(text)

				select {
				case out <- reading:
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return out, nil
}

// drainAllFrames returns the current frame plus all queued ones in order, keeping
// at most the most recent maxBatch.
func drainAllFrames(frames <-chan video.Frame, current video.Frame, maxBatch int) []video.Frame {
	batch := []video.Frame{current}
	for {
		select {
		case f, ok := <-frames:
			if !ok {
				return capTail(batch, maxBatch)
			}
			batch = append(batch, f)
		default:
			return capTail(batch, maxBatch)
		}
	}
}

// capTail returns at most max trailing elements of b (the most recent frames).
func capTail(b []video.Frame, max int) []video.Frame {
	if max > 0 && len(b) > max {
		return b[len(b)-max:]
	}
	return b
}

// debugRecord is one verdicts.jsonl line: a batch's verdict, timing, and JPEG names.
type debugRecord struct {
	Time        string   `json:"time"`          // wall-clock time the verdict returned
	UnixMs      int64    `json:"unix_ms"`       // same, epoch milliseconds (sortable)
	Verdict     string   `json:"verdict"`       // raw text the VLM returned
	CallMs      int64    `json:"call_ms"`       // VLM round-trip
	NewestAgeMs int64    `json:"newest_age_ms"` // age of freshest frame at verdict time
	WindowMs    int64    `json:"window_ms"`     // span of time the batch covers
	Frames      []string `json:"frames"`        // JPEG filenames, oldest -> newest
}

// dumpDebug writes the batch frames and appends its verdict to verdicts.jsonl.
func (s *laydownDetector) dumpDebug(batch []video.Frame, verdict string, callMs, newestAgeMs, windowMs int64) {
	names := make([]string, 0, len(batch))
	for i, f := range batch {
		name := fmt.Sprintf("frame_%d_%d.jpg", f.Timestamp.UnixMilli(), i)
		if err := os.WriteFile(filepath.Join(s.debugDir, name), f.JPEG, 0o644); err != nil {
			s.log.Warn("image dump: write frame failed", zap.Error(err))
			continue
		}
		names = append(names, name)
	}

	now := time.Now()
	line, err := json.Marshal(debugRecord{
		Time:        now.Format(time.RFC3339Nano),
		UnixMs:      now.UnixMilli(),
		Verdict:     verdict,
		CallMs:      callMs,
		NewestAgeMs: newestAgeMs,
		WindowMs:    windowMs,
		Frames:      names,
	})
	if err != nil {
		return
	}

	f, err := os.OpenFile(filepath.Join(s.debugDir, "verdicts.jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		s.log.Warn("image dump: open verdicts.jsonl failed", zap.Error(err))
		return
	}
	defer func() { _ = f.Close() }()
	_, _ = f.Write(append(line, '\n'))
}

// classify applies the latch+debounce: ALERT asserts immediately, clearing requires
// clearStreak consecutive non-alert frames.
func (s *laydownDetector) classify(text string) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if strings.Contains(strings.ToUpper(text), "ALERT") {
		if !s.alerted {
			s.alertedAt = time.Now()
		}
		s.alerted = true
		s.clearCount = 0
		s.lastAlert = text
		tts.Priority.Store(true)
		return text
	}

	if s.alerted {
		s.clearCount++
		held := time.Since(s.alertedAt)

		// Clear only when both the clear streak and the min-hold time are met.
		streakMet := s.clearCount >= s.clearStreak
		holdMet := s.minHold <= 0 || held >= s.minHold
		if !streakMet || !holdMet {
			s.log.Debug("holding alert",
				zap.Int("clear_count", s.clearCount),
				zap.Int("clear_streak", s.clearStreak),
				zap.Duration("held", held),
				zap.Duration("min_hold", s.minHold),
			)
			return s.lastAlert
		}

		s.alerted = false
		s.clearCount = 0
		tts.Priority.Store(false)
		s.log.Info("alert cleared",
			zap.Int("clear_streak", s.clearStreak),
			zap.Duration("held", held),
		)
	}

	return text
}

func (s *laydownDetector) Poll(context.Context) (any, error) {
	return nil, nil
}

// RawToText records the latest verdict without clearing it, so it persists across ticks.
func (s *laydownDetector) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)
	s.mu.Lock()
	s.latest = *msg
	s.hasLatest = true
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the latched verdict with the "Vision:" prefix.
func (s *laydownDetector) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.hasLatest {
		return ""
	}

	result := fmt.Sprintf("\n%s: '%s'\n", vlmDescriptor, s.latest.Message)

	ts := time.Unix(0, int64(s.latest.Timestamp*1e9))
	providers.IO().AddInput(s.name, s.latest.Message, ts)

	return result
}

// TriggersTick wakes the cortex as soon as a fresh verdict arrives.
func (s *laydownDetector) TriggersTick() bool {
	return true
}

func (s *laydownDetector) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	cancel := s.cancel
	s.mu.Unlock()

	tts.Priority.Store(false)
	if cancel != nil {
		cancel()
	}
	s.source.Stop()
	s.log.Info("stopping sensor")
}
