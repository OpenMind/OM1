package vlm

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
	video "github.com/openmind/om1/internal/providers/vlm"
)

func init() {
	inputs.Register("VLMBackground", NewVLMBackground)
}

const (
	vlmBackgroundName             = "VLMBackground"
	vlmBackgroundDefaultPollSec   = 0.5
	vlmBackgroundDefaultMaxAgeSec = 0.0
)

type VLMBackgroundConfig struct {
	PollIntervalSec float64 `json:"poll_interval_sec"`
	MaxAgeSec       float64 `json:"max_age_sec"`
}

// vlmBackgroundSensor is a passive VLM input that surfaces the scene description.
type vlmBackgroundSensor struct {
	log    *zap.Logger
	period time.Duration
	maxAge time.Duration

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

// NewVLMBackground constructs a sensor that reads the latest description published by a VLM global background.
func NewVLMBackground(configMap map[string]any) (inputs.Sensor, error) {
	var cfg VLMBackgroundConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}
	if cfg.PollIntervalSec <= 0 {
		cfg.PollIntervalSec = vlmBackgroundDefaultPollSec
	}
	if cfg.MaxAgeSec < 0 {
		cfg.MaxAgeSec = vlmBackgroundDefaultMaxAgeSec
	}

	log := logger.Get().Named(vlmBackgroundName)
	log.Info("initializing",
		zap.Float64("poll_interval_sec", cfg.PollIntervalSec),
		zap.Float64("max_age_sec", cfg.MaxAgeSec),
	)

	return &vlmBackgroundSensor{
		log:    log,
		period: time.Duration(cfg.PollIntervalSec * float64(time.Second)),
		maxAge: time.Duration(cfg.MaxAgeSec * float64(time.Second)),
	}, nil
}

// read returns the latest description, honoring the configured staleness window.
func (s *vlmBackgroundSensor) read() (string, time.Time, bool) {
	if s.maxAge > 0 {
		return video.LatestDescription().GetFresh(s.maxAge)
	}
	return video.LatestDescription().Get()
}

// Listen polls the background description and emits each new one as it appears.
func (s *vlmBackgroundSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(s.period)
		defer ticker.Stop()

		var lastTS time.Time
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}

			text, ts, ok := s.read()
			if !ok || !ts.After(lastTS) {
				continue
			}
			lastTS = ts

			s.append(text)
			providers.IO().AddInput(vlmBackgroundName, text, ts)
		}
	}()
	return out, nil
}

func (s *vlmBackgroundSensor) Poll(context.Context) (any, error) {
	text, _, ok := s.read()
	if !ok {
		return nil, nil
	}
	return text, nil
}

// RawToText buffers a description string. Defensive: Listen is passive.
func (s *vlmBackgroundSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}
	msg := inputs.NewMessage(text)
	s.append(text)
	return msg, nil
}

// append adds a description to the bounded message buffer.
func (s *vlmBackgroundSensor) append(text string) {
	msg := inputs.NewMessage(text)
	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > vlmMaxMessages {
		s.messages = s.messages[len(s.messages)-vlmMaxMessages:]
	}
	s.mu.Unlock()
}

// FormattedLatestBuffer returns the newest description and clears the buffer.
func (s *vlmBackgroundSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", vlmDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(vlmBackgroundName, latest.Message, ts)
	s.messages = nil

	return result
}

func (s *vlmBackgroundSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
}
