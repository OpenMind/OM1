package inputs

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
	"github.com/openmind/om1/internal/util"
)

func init() {
	inputs.Register("FacePresence", NewFacePresence)
}

const (
	facePresenceDescriptor  = "Face Presence Sensor"
	facePresenceIOKey       = "FacePresence"
	facePresenceMaxMessages = 10
)

// FacePresenceConfig is the JSON configuration for the FacePresence sensor.
type FacePresenceConfig struct {
	// BaseURL is the base URL for the face HTTP service.
	BaseURL string `json:"face_http_base_url"`
	// RecentSec is the time window in seconds used to consider a face present.
	RecentSec float64 `json:"face_recent_sec"`
	// PollIntervalSec is the interval in seconds between successive polls.
	PollIntervalSec float64 `json:"face_poll_interval_sec"`
}

// FacePresenceSensor implements the inputs.Sensor interface for face presence data.
type FacePresenceSensor struct {
	cfg      FacePresenceConfig
	log      *zap.Logger
	provider *providers.FacePresenceProvider
	period   time.Duration

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

// NewFacePresence constructs a FacePresenceSensor from the decoded config map.
func NewFacePresence(configMap map[string]any) (inputs.Sensor, error) {
	var cfg FacePresenceConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://127.0.0.1:6793"
	}
	if cfg.RecentSec <= 0 {
		cfg.RecentSec = 1.0
	}
	if cfg.PollIntervalSec <= 0 {
		cfg.PollIntervalSec = 0.2
	}

	log := logger.Get().Named("FacePresence")
	provider := providers.NewFacePresenceProvider(providers.FacePresenceConfig{
		BaseURL:   cfg.BaseURL,
		RecentSec: cfg.RecentSec,
		Timeout:   2 * time.Second,
	})

	log.Info("initializing",
		zap.String("base_url", cfg.BaseURL),
		zap.Float64("recent_sec", cfg.RecentSec),
		zap.Float64("poll_interval_sec", cfg.PollIntervalSec),
	)

	return &FacePresenceSensor{
		cfg:      cfg,
		log:      log,
		provider: provider,
		period:   time.Duration(cfg.PollIntervalSec * float64(time.Second)),
	}, nil
}

// Listen polls the face-presence service at the configured cadence and yields
// each formatted presence line on the returned channel until ctx is cancelled.
func (s *FacePresenceSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(s.period)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			raw, err := s.Poll(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				s.log.Warn("failed to fetch snapshot", zap.Error(err))
				util.Sleep(ctx, 2*time.Second)
				continue
			}

			select {
			case out <- raw:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out, nil
}

// Poll fetches a single presence snapshot and returns its formatted text line.
func (s *FacePresenceSensor) Poll(ctx context.Context) (any, error) {
	snap, err := s.provider.FetchSnapshot(ctx)
	if err != nil {
		return nil, err
	}
	return snap.ToText(), nil
}

// RawToText converts a raw presence line into a timestamped Message and appends
// it to the bounded in-memory history.
func (s *FacePresenceSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > facePresenceMaxMessages {
		s.messages = s.messages[len(s.messages)-facePresenceMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the newest presence line as a compact,
// prompt-ready block and clears the history. It returns "" when empty.
func (s *FacePresenceSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", facePresenceDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(facePresenceIOKey, latest.Message, ts)
	s.messages = nil

	return result
}

// Stop marks the sensor stopped. The polling goroutine terminates via context
// cancellation in Listen.
func (s *FacePresenceSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
}
