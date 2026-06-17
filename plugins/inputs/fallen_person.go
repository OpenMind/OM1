package inputs

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

	// fallenDefaultDebugPeriod throttles debug dumps when debug_dir is set, so a fast
	// poll loop does not flood the disk with frames.
	fallenDefaultDebugPeriod = time.Second
)

type fallenPersonConfig struct {
	BaseURL        string  `json:"base_url"`
	Path           string  `json:"path"`
	PollPeriodMS   int     `json:"poll_period_ms"`
	TimeoutMS      int     `json:"timeout_ms"`
	ClearStreak    int     `json:"clear_streak"`
	MinHoldSeconds float64 `json:"min_hold_seconds"`
	LockWidthFrac  float64 `json:"lock_width_frac"`

	// DebugDir, when set, enables dumping each polled frame (decoded from frame_b64)
	// and its analysis to that directory. DebugPeriodMS throttles the dump rate.
	DebugDir      string `json:"debug_dir"`
	DebugPeriodMS int    `json:"debug_period_ms"`
}

type fallenPersonTracker struct {
	log         *zap.Logger
	provider    *providers.FallenPersonProvider
	pollPeriod  time.Duration
	clearStreak int
	minHold     time.Duration
	lockWidth   float64

	debugDir    string
	debugPeriod time.Duration
	lastDebugAt time.Time

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

	debugPeriod := fallenDefaultDebugPeriod
	if cfg.DebugPeriodMS > 0 {
		debugPeriod = time.Duration(cfg.DebugPeriodMS) * time.Millisecond
	}
	log := logger.Get().Named("FallenPersonTracker")
	if cfg.DebugDir != "" {
		if err := os.MkdirAll(cfg.DebugDir, 0o755); err != nil {
			log.Warn("debug dump disabled: cannot create debug_dir", zap.String("dir", cfg.DebugDir), zap.Error(err))
			cfg.DebugDir = ""
		}
	}
	log.Info("initializing",
		zap.String("base_url", cfg.BaseURL),
		zap.String("path", cfg.Path),
		zap.Duration("poll_period", pollPeriod),
		zap.Int("clear_streak", clearStreak),
		zap.Float64("min_hold_seconds", minHoldSeconds),
		zap.Float64("lock_width_frac", lockWidth),
		zap.String("debug_dir", cfg.DebugDir),
	)

	return &fallenPersonTracker{
		log: log,
		provider: providers.NewFallenPersonProvider(providers.FallenPersonConfig{
			BaseURL:     cfg.BaseURL,
			Path:        cfg.Path,
			Timeout:     time.Duration(cfg.TimeoutMS) * time.Millisecond,
			CacheFrames: clearStreak,
		}),
		pollPeriod:  pollPeriod,
		clearStreak: clearStreak,
		minHold:     time.Duration(minHoldSeconds * float64(time.Second)),
		lockWidth:   lockWidth,
		debugDir:    cfg.DebugDir,
		debugPeriod: debugPeriod,
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
			s.maybeDumpDebug(snap, reading)
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

// fallenTarget is the chosen closest detection's geometry, embedded in a debug record.
// Location/Distance mirror the human-readable verdict ("Location in view: <location>.
// Distance: <distance>.") so the analysis is easy to scan.
type fallenTarget struct {
	Name        string    `json:"name"`
	Location    string    `json:"location"` // left / center / right
	Distance    string    `json:"distance"` // near / far
	NormErrX    float64   `json:"norm_err_x"`
	WidthFrac   float64   `json:"width_frac"`
	Confidence  float64   `json:"confidence"`
	FaceMatched bool      `json:"face_matched"` // true when centering used the face bbox
	BodyBbox    []float64 `json:"body_bbox,omitempty"`
	FaceBbox    []float64 `json:"face_bbox,omitempty"`
}

// fallenDebugRecord is one verdicts.jsonl line: the analysis for a polled frame plus
// the name of the image file written alongside it. Verdict is the exact text emitted
// to the cortex this tick (e.g. "ALERT: ... Location in view: center. Distance: far.").
type fallenDebugRecord struct {
	Time       string                      `json:"time"`
	UnixMs     int64                       `json:"unix_ms"`
	Verdict    string                      `json:"verdict"`
	Alert      bool                        `json:"alert"`
	Present    bool                        `json:"present"`
	Cached     bool                        `json:"cached"`
	Target     *fallenTarget               `json:"target,omitempty"`
	Detections []providers.FallenDetection `json:"detections"`
	FrameW     float64                     `json:"frame_w"`
	Frame      string                      `json:"frame,omitempty"`
}

// maybeDumpDebug writes the decoded frame and its analysis (including the emitted
// verdict) when debug dumping is enabled, throttled to debugPeriod. It runs only on
// the poll goroutine.
func (s *fallenPersonTracker) maybeDumpDebug(snap providers.FallenSnapshot, verdict string) {
	if s.debugDir == "" {
		return
	}
	now := time.Now()
	if s.debugPeriod > 0 && now.Sub(s.lastDebugAt) < s.debugPeriod {
		return
	}
	s.lastDebugAt = now

	frameName := ""
	if snap.FrameB64 != "" {
		if img, ext, err := decodeFrame(snap.FrameB64); err != nil {
			s.log.Warn("debug dump: decode frame_b64 failed", zap.Error(err))
		} else {
			frameName = fmt.Sprintf("frame_%d%s", now.UnixMilli(), ext)
			if err := os.WriteFile(filepath.Join(s.debugDir, frameName), img, 0o644); err != nil {
				s.log.Warn("debug dump: write frame failed", zap.Error(err))
				frameName = ""
			}
		}
	}

	rec := fallenDebugRecord{
		Time:       now.Format(time.RFC3339Nano),
		UnixMs:     now.UnixMilli(),
		Verdict:    verdict,
		Alert:      snap.Alert,
		Present:    snap.Present,
		Cached:     snap.Cached,
		Detections: snap.Detections,
		FrameW:     snap.FrameW,
		Frame:      frameName,
	}
	if snap.Present {
		distance := "far"
		if snap.WidthFrac >= s.lockWidth {
			distance = "near"
		}
		rec.Target = &fallenTarget{
			Name:        snap.Name,
			Location:    snap.HPos,
			Distance:    distance,
			NormErrX:    snap.NormErrX,
			WidthFrac:   snap.WidthFrac,
			Confidence:  snap.Confidence,
			FaceMatched: snap.FaceBbox != nil,
			BodyBbox:    snap.BodyBbox,
			FaceBbox:    snap.FaceBbox,
		}
	}

	line, err := json.Marshal(rec)
	if err != nil {
		return
	}
	f, err := os.OpenFile(filepath.Join(s.debugDir, "verdicts.jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		s.log.Warn("debug dump: open verdicts.jsonl failed", zap.Error(err))
		return
	}
	defer func() { _ = f.Close() }()
	_, _ = f.Write(append(line, '\n'))
}

// decodeFrame decodes a base64 image, tolerating a "data:image/...;base64," prefix,
// and returns the bytes plus a file extension guessed from the data-URL mime or the
// image's magic bytes.
func decodeFrame(b64 string) ([]byte, string, error) {
	ext := ".jpg"
	if strings.HasPrefix(b64, "data:") {
		if i := strings.Index(b64, ","); i >= 0 {
			if strings.Contains(b64[:i], "image/png") {
				ext = ".png"
			}
			b64 = b64[i+1:]
		}
	}
	img, err := base64.StdEncoding.DecodeString(strings.TrimSpace(b64))
	if err != nil {
		return nil, "", err
	}
	switch {
	case len(img) >= 4 && img[0] == 0x89 && img[1] == 'P' && img[2] == 'N' && img[3] == 'G':
		ext = ".png"
	case len(img) >= 2 && img[0] == 0xFF && img[1] == 0xD8:
		ext = ".jpg"
	}
	return img, ext, nil
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
