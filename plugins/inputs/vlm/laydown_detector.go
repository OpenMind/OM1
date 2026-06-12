package vlm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	video "github.com/openmind/om1/internal/providers/vlm"
)

// PersonLaydownDetector is an RTSP+VLM input specialised for spotting a person
// lying on the ground. It differs from the generic VLM sensor in three ways that
// matter for a safety reaction:
//
//  1. Latching with debounce — an ALERT is asserted on the first detection and
//     held until `clear_streak` CONSECUTIVE clear frames arrive. A single noisy
//     "no person" frame (the VLM flip-flops, and slow calls return stale frames)
//     can no longer bounce the robot back into patrol.
//  2. Freshest-frame-only — the backlog of frames queued behind a slow VLM call
//     is drained and only the most recent frame is analysed, so a clear verdict
//     can't come from a seconds-old pre-collapse frame.
//  3. Persistent buffer — the latest latched verdict is reported on EVERY cortex
//     tick (not cleared after one read), so there are no "no Vision line" gaps
//     for the cortex to misread as "all clear".
//
// It registers under the type name "PersonLaydownDetector" and is a drop-in
// replacement for VLMGeminiRTSP in a config: same "Vision:" prefix and same
// ALERT / "No person lying on the ground." vocabulary.
func init() {
	inputs.Register("PersonLaydownDetector", NewPersonLaydownDetector)
}

const defaultClearStreak = 3

// laydownBufferSize is the frame-channel depth. It must comfortably exceed
// fps * worst-case-VLM-call-seconds so frames produced during a slow call are
// kept (not dropped) and can be batched into the next request.
const laydownBufferSize = 16

// defaultMaxBatch caps how many frames are sent in a single multi-image VLM
// request. It bounds per-call latency and token cost while still covering every
// frame captured during the previous (~2-3s) call at a modest fps.
const defaultMaxBatch = 6

// laydownDefaults mirrors geminiDefaults but ships a detection-focused prompt and
// a small token budget (the reply is a single line).
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
	log.Info("initializing",
		zap.String("rtsp_url", cfg.RTSPURL),
		zap.String("model", cfg.Model),
		zap.Int("fps", cfg.FPS),
		zap.Int("clear_streak", clearStreak),
		zap.Duration("min_hold", minHold),
		zap.Int("max_frames_per_call", maxBatch),
	)

	source := video.NewVideoRTSPStream(video.VideoRTSPStreamConfig{
		RTSPURL:     cfg.RTSPURL,
		FPS:         cfg.FPS,
		Width:       cfg.Width,
		Height:      cfg.Height,
		JPEGQuality: cfg.JPEGQuality,
		// The default 1-slot buffer drops the NEWEST frame while a (~2-3s) VLM
		// call is in flight, so the next frame we read is already ~one call-length
		// stale — that's why frame_age_ms runs ~2x call_ms. A deeper buffer lets
		// frames produced during the call accumulate so drainLatestFrame can pick
		// the freshest one, cutting capture-to-verdict lag to ~1x the call time.
		BufferSize: laydownBufferSize,
	})

	return &laydownDetector{
		name:        "PersonLaydownDetector",
		log:         log,
		source:      source,
		clearStreak: clearStreak,
		minHold:     minHold,
		maxBatch:    maxBatch,
		describer: video.NewDescriber(video.Describer{
			Name:      "PersonLaydownDetector",
			APIKey:    cfg.APIKey,
			BaseURL:   cfg.BaseURL,
			Model:     cfg.Model,
			Prompt:    cfg.Prompt,
			MaxTokens: cfg.MaxTokens,
			Log:       log,
		}),
	}, nil
}

// Listen captures frames, analyses only the freshest one per VLM call, and emits
// the latched verdict.
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

				// Batch the backlog instead of dropping it: every frame captured
				// while the previous call was in flight is sent together in one
				// multi-image request, so a person who only appears in a frame
				// between calls (e.g. during a walk-by) can no longer slip through.
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

				// call_ms = VLM round-trip; newest_age_ms = capture-to-verdict lag
				// of the freshest frame (the true reaction latency); window_ms = the
				// span of time the batch covers (0 when only one frame was queued).
				s.log.Info("vlm latency",
					zap.Int("frames", len(batch)),
					zap.Int64("call_ms", callLatency.Milliseconds()),
					zap.Int64("newest_age_ms", time.Since(newest.Timestamp).Milliseconds()),
					zap.Int64("window_ms", newest.Timestamp.Sub(oldest.Timestamp).Milliseconds()),
					zap.String("verdict", text),
				)

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

// drainAllFrames collects the current frame plus every other frame already
// queued, returning them in chronological order. When more than maxBatch frames
// are available it keeps the most recent maxBatch (the freshest views, and a
// bound on per-call latency and token cost).
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

// classify applies the latch+debounce and returns the verdict text to surface to
// the cortex. ALERT asserts immediately; clearing requires clearStreak
// consecutive non-alert frames.
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
		return text
	}

	if s.alerted {
		s.clearCount++
		held := time.Since(s.alertedAt)

		// Clear only when BOTH conditions hold: enough consecutive clear frames
		// AND the alert has been latched for at least minHold wall-clock time.
		// The time floor guarantees a visible, predictable stop duration even
		// when clear frames arrive in a fast burst.
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

// RawToText records the latest verdict. It does not clear the buffer, so the
// current latched state is reported on every cortex tick.
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

// FormattedLatestBuffer returns the current latched verdict, prefixed like the
// generic VLM sensor so existing prompts ("Vision:") keep working. It does not
// clear the buffer.
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

// TriggersTick wakes the cortex as soon as a fresh verdict arrives instead of
// waiting for the next periodic tick.
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

	if cancel != nil {
		cancel()
	}
	s.source.Stop()
	s.log.Info("stopping sensor")
}
