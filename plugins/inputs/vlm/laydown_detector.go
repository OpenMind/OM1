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

// laydownDefaults mirrors geminiDefaults but ships a detection-focused prompt and
// a small token budget (the reply is a single line).
var laydownDefaults = providerDefaults{
	baseURL: geminiDefaults.baseURL,
	model:   geminiDefaults.model,
	prompt: "You are the safety vision system of a patrol robot. Look carefully at this image. " +
		"Decide whether a PERSON is lying on the ground or floor — collapsed, fallen, slumped, or " +
		"otherwise not standing or sitting upright in a normal way. Respond in ONE line. If such a " +
		"person is present, respond exactly in this form: 'ALERT: a person is lying on the ground. " +
		"<brief description of their position and any visible signs of a medical condition>.' If no one " +
		"is lying on the ground, respond exactly: 'No person lying on the ground.' Do not explain your reasoning.",
	maxTokens: 64,
}

type laydownDetector struct {
	name        string
	log         *zap.Logger
	describer   *video.Describer
	source      frameSource
	clearStreak int

	mu         sync.Mutex
	latest     inputs.Message
	hasLatest  bool
	alerted    bool
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
		ClearStreak int `json:"clear_streak"`
	}
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &extra)
	}
	clearStreak := extra.ClearStreak
	if clearStreak <= 0 {
		clearStreak = defaultClearStreak
	}

	log := logger.Get().Named("PersonLaydownDetector")
	log.Info("initializing",
		zap.String("rtsp_url", cfg.RTSPURL),
		zap.String("model", cfg.Model),
		zap.Int("fps", cfg.FPS),
		zap.Int("clear_streak", clearStreak),
	)

	source := video.NewVideoRTSPStream(video.VideoRTSPStreamConfig{
		RTSPURL:     cfg.RTSPURL,
		FPS:         cfg.FPS,
		Width:       cfg.Width,
		Height:      cfg.Height,
		JPEGQuality: cfg.JPEGQuality,
	})

	return &laydownDetector{
		name:        "PersonLaydownDetector",
		log:         log,
		source:      source,
		clearStreak: clearStreak,
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

				// Drop the backlog: analyse only the most recent frame so a slow
				// VLM call can't make us classify a stale, pre-collapse image.
				frame = drainLatestFrame(frames, frame)

				providers.LatestFrame().Set(frame.JPEG, frame.Timestamp)
				text, err := s.describer.Describe(ctx, base64.StdEncoding.EncodeToString(frame.JPEG))
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					s.log.Warn("vision request failed", zap.Error(err))
					continue
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

// drainLatestFrame returns the newest frame currently available on the channel,
// discarding any older queued frames.
func drainLatestFrame(frames <-chan video.Frame, current video.Frame) video.Frame {
	for {
		select {
		case f, ok := <-frames:
			if !ok {
				return current
			}
			current = f
		default:
			return current
		}
	}
}

// classify applies the latch+debounce and returns the verdict text to surface to
// the cortex. ALERT asserts immediately; clearing requires clearStreak
// consecutive non-alert frames.
func (s *laydownDetector) classify(text string) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if strings.Contains(strings.ToUpper(text), "ALERT") {
		s.alerted = true
		s.clearCount = 0
		s.lastAlert = text
		return text
	}

	if s.alerted {
		s.clearCount++
		if s.clearCount < s.clearStreak {
			// Debounced: hold the alert, ignore this lone clear frame.
			s.log.Debug("holding alert through clear frame",
				zap.Int("clear_count", s.clearCount),
				zap.Int("clear_streak", s.clearStreak),
			)
			return s.lastAlert
		}
		s.alerted = false
		s.clearCount = 0
		s.log.Info("alert cleared after consecutive clear frames", zap.Int("clear_streak", s.clearStreak))
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
