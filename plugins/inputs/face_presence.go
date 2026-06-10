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

	facePresenceDefaultBaseURL         = "http://127.0.0.1:6793"
	facePresenceDefaultRecentSec       = 1.0
	facePresenceDefaultPollIntervalSec = 0.2
)

type FacePresenceConfig struct {
	BaseURL         string  `json:"face_http_base_url"`
	RecentSec       float64 `json:"face_recent_sec"`
	PollIntervalSec float64 `json:"face_poll_interval_sec"`
}

type FacePresenceSensor struct {
	cfg      FacePresenceConfig
	log      *zap.Logger
	provider *providers.FacePresenceProvider
	period   time.Duration

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

func NewFacePresence(configMap map[string]any) (inputs.Sensor, error) {
	var cfg FacePresenceConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = facePresenceDefaultBaseURL
	}
	if cfg.RecentSec <= 0 {
		cfg.RecentSec = facePresenceDefaultRecentSec
	}
	if cfg.PollIntervalSec <= 0 {
		cfg.PollIntervalSec = facePresenceDefaultPollIntervalSec
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

// Listen polls face presence and updates state.
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

			snap, err := s.provider.FetchSnapshot(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				s.log.Warn("failed to fetch snapshot", zap.Error(err))
				util.Sleep(ctx, 2*time.Second)
				continue
			}

			text := snap.ToText()
			if text == "" {
				continue
			}

			msg := inputs.NewMessage(text)
			s.mu.Lock()
			s.messages = append(s.messages, *msg)
			if len(s.messages) > facePresenceMaxMessages {
				s.messages = s.messages[len(s.messages)-facePresenceMaxMessages:]
			}
			s.mu.Unlock()

			// Refresh shared IO entry and dynamic vars.
			providers.IO().AddInput(facePresenceIOKey, text, time.Now())
			providers.IO().SetDynamicVar("current_user_id", snap.ClosestUUID)
			providers.IO().SetDynamicVar("current_user_name", snap.ClosestName)
		}
	}()
	return out, nil
}

func (s *FacePresenceSensor) Poll(ctx context.Context) (any, error) {
	snap, err := s.provider.FetchSnapshot(ctx)
	if err != nil {
		return nil, err
	}
	return snap.ToText(), nil
}

// RawToText implements inputs.Sensor. Defensive no-op since Listen is passive.
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

// FormattedLatestBuffer returns the newest presence line and clears the buffer.
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
