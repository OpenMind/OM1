package inputs

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/providers/tts"
)

// FallenPersonTracker is an HTTP-polled input that replaces the slow Gemini VLM
// laydown detector: it reads bbox detections from a localhost endpoint, latches the
// downed-person ALERT with debounce, and publishes the closest target's geometry so
// the motion connector can approach it geometrically (no LLM in the motion loop).
func init() {
	inputs.Register("FallenPersonTracker", NewFallenPersonTracker)
}

const (
	fallenDescriptor            = "Vision"
	fallenDefaultPollPeriod     = 100 * time.Millisecond
	fallenDefaultClearStreak    = 2
	fallenDefaultMinHoldSeconds = 5.0
	fallenNoPersonText          = "No person lying on the ground."
)

type fallenPersonConfig struct {
	BaseURL        string  `json:"base_url"`
	Path           string  `json:"path"`
	PollPeriodMS   int     `json:"poll_period_ms"`
	TimeoutMS      int     `json:"timeout_ms"`
	ClearStreak    int     `json:"clear_streak"`
	MinHoldSeconds float64 `json:"min_hold_seconds"`
	LockWidthFrac  float64 `json:"lock_width_frac"`
}

type fallenPersonTracker struct {
	log         *zap.Logger
	provider    *providers.FallenPersonProvider
	pollPeriod  time.Duration
	clearStreak int
	minHold     time.Duration
	lockWidth   float64

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

// NewFallenPersonTracker builds the HTTP bbox-based downed-person input.
func NewFallenPersonTracker(configMap map[string]any) (inputs.Sensor, error) {
	var cfg fallenPersonConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	pollPeriod := fallenDefaultPollPeriod
	if cfg.PollPeriodMS > 0 {
		pollPeriod = time.Duration(cfg.PollPeriodMS) * time.Millisecond
	}
	clearStreak := cfg.ClearStreak
	if clearStreak <= 0 {
		clearStreak = fallenDefaultClearStreak
	}
	minHoldSeconds := cfg.MinHoldSeconds
	if minHoldSeconds <= 0 {
		minHoldSeconds = fallenDefaultMinHoldSeconds
	}
	lockWidth := cfg.LockWidthFrac
	if lockWidth <= 0 {
		lockWidth = providers.DefaultLockWidthFrac
	}

	log := logger.Get().Named("FallenPersonTracker")
	log.Info("initializing",
		zap.String("base_url", cfg.BaseURL),
		zap.String("path", cfg.Path),
		zap.Duration("poll_period", pollPeriod),
		zap.Int("clear_streak", clearStreak),
		zap.Float64("min_hold_seconds", minHoldSeconds),
		zap.Float64("lock_width_frac", lockWidth),
	)

	return &fallenPersonTracker{
		log: log,
		provider: providers.NewFallenPersonProvider(providers.FallenPersonConfig{
			BaseURL: cfg.BaseURL,
			Path:    cfg.Path,
			Timeout: time.Duration(cfg.TimeoutMS) * time.Millisecond,
		}),
		pollPeriod:  pollPeriod,
		clearStreak: clearStreak,
		minHold:     time.Duration(minHoldSeconds * float64(time.Second)),
		lockWidth:   lockWidth,
	}, nil
}

// Listen polls the endpoint and emits the latched verdict on every fresh reading.
func (s *fallenPersonTracker) Listen(ctx context.Context) (<-chan any, error) {
	ctx, cancel := context.WithCancel(ctx)
	s.mu.Lock()
	s.cancel = cancel
	s.mu.Unlock()

	out := make(chan any)

	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(s.pollPeriod)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			snap, err := s.provider.FetchSnapshot(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				s.log.Debug("fallen poll failed", zap.Error(err))
				continue
			}

			reading := s.classify(snap)
			if reading == "" {
				continue
			}

			select {
			case out <- reading:
			case <-ctx.Done():
				return
			}
		}
	}()

	return out, nil
}

// classify applies the latch+debounce and publishes the target geometry: an ALERT
// asserts immediately; clearing requires clearStreak consecutive clear readings and
// the minimum hold. It mirrors the old VLM detector's semantics so downstream
// speak/emotion behaviour is unchanged.
func (s *fallenPersonTracker) classify(snap providers.FallenSnapshot) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	alert := snap.Alert && snap.Present

	if alert {
		providers.SetFallenTarget(providers.FallenTarget{
			Present:    true,
			NormErrX:   snap.NormErrX,
			WidthFrac:  snap.WidthFrac,
			Confidence: snap.Confidence,
			Name:       snap.Name,
		})

		if !s.alerted {
			s.alertedAt = time.Now()
		}
		s.alerted = true
		s.clearCount = 0
		s.lastAlert = s.alertText(snap)
		tts.Priority.Store(true)
		providers.SetPersonDownAlert(true)
		return s.lastAlert
	}

	if s.alerted {
		s.clearCount++
		held := time.Since(s.alertedAt)

		streakMet := s.clearCount >= s.clearStreak
		holdMet := s.minHold <= 0 || held >= s.minHold
		if !streakMet || !holdMet {
			return s.lastAlert
		}

		s.alerted = false
		s.clearCount = 0
		tts.Priority.Store(false)
		providers.SetPersonDownAlert(false)
		providers.SetPersonDownArrived(false)
		providers.SetFallenTarget(providers.FallenTarget{})
		tts.RequestInterrupt()
		s.log.Info("alert cleared", zap.Int("clear_streak", s.clearStreak), zap.Duration("held", held))
	}

	return fallenNoPersonText
}

// alertText synthesizes a verdict line compatible with the existing rescue prompt so
// the LLM keeps speaking and emoting; movement is driven separately from the geometry.
func (s *fallenPersonTracker) alertText(snap providers.FallenSnapshot) string {
	distance := "far"
	if snap.WidthFrac >= s.lockWidth {
		distance = "near"
	}
	who := "a person"
	if name := strings.TrimSpace(snap.Name); name != "" {
		who = name
	}
	return fmt.Sprintf(
		"ALERT: a person is lying on the ground (%s). Location in view: %s. Distance: %s.",
		who, snap.HPos, distance,
	)
}

func (s *fallenPersonTracker) Poll(context.Context) (any, error) { return nil, nil }

// RawToText records the latest verdict so it persists across cortex ticks.
func (s *fallenPersonTracker) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)
	s.mu.Lock()
	s.latest = *msg
	s.hasLatest = true
	s.mu.Unlock()

	providers.BumpVisionSeq()

	return msg, nil
}

// FormattedLatestBuffer returns the latched verdict with the "Vision:" prefix.
func (s *fallenPersonTracker) FormattedLatestBuffer() string {
	s.mu.Lock()
	hasLatest := s.hasLatest
	latest := s.latest
	s.mu.Unlock()

	if !hasLatest {
		return ""
	}

	result := fmt.Sprintf("\n%s: '%s'\n", fallenDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(fallenDescriptor, latest.Message, ts)

	return result
}

// TriggersTick wakes the cortex as soon as a fresh verdict arrives.
func (s *fallenPersonTracker) TriggersTick() bool { return true }

func (s *fallenPersonTracker) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	cancel := s.cancel
	s.mu.Unlock()

	tts.Priority.Store(false)
	providers.SetPersonDownAlert(false)
	providers.SetPersonDownArrived(false)
	providers.SetFallenTarget(providers.FallenTarget{})
	if cancel != nil {
		cancel()
	}
	s.log.Info("stopping sensor")
}
