package go2

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
	odomprovider "github.com/openmind/om1/internal/providers/unitree/go2"
)

func init() {
	inputs.Register("UnitreeGo2Odom", NewUnitreeGo2Odom)
}

const (
	odomDescriptor   = "Information about your location and body pose, to help plan your movements."
	odomPollPeriod   = 100 * time.Millisecond
	odomMaxMessages  = 10
	odomSittingText  = "You are sitting down - do not generate new movement commands. "
	odomMovingText   = "You are moving - do not generate new movement commands. "
	odomStandingText = "You are standing still - you can move if you want to. "
)

type OdomConfig struct {
	Topic string `json:"topic"`
}

type OdomSensor struct {
	log      *zap.Logger
	provider *odomprovider.OdomZenohProvider

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

// NewUnitreeGo2Odom creates a new OdomSensor with the given configuration.
func NewUnitreeGo2Odom(configMap map[string]any) (inputs.Sensor, error) {
	var cfg OdomConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	log := logger.Get()
	log.Info("UnitreeGo2Odom: initializing", zap.String("topic", cfg.Topic))

	return &OdomSensor{
		log:      log,
		provider: odomprovider.NewOdomZenohProvider(cfg.Topic),
	}, nil
}

// Listen starts a goroutine that polls the odometry provider at a fixed interval and sends snapshots to the returned channel.
func (s *OdomSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(odomPollPeriod)
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

// Poll returns the latest odometry snapshot from the provider.
func (s *OdomSensor) Poll(_ context.Context) (any, error) {
	return s.provider.Position(), nil
}

// RawToText converts an odometry snapshot to a human-readable message about the robot's current state.
func (s *OdomSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	pos, ok := raw.(odomprovider.OdomPosition)
	if !ok {
		return nil, nil
	}

	var res string
	switch {
	case pos.BodyAttitude == odomprovider.RobotStateSitting:
		res = odomSittingText
	case pos.Moving:
		res = odomMovingText
	default:
		res = odomStandingText
	}

	msg := inputs.NewMessage(res)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > odomMaxMessages {
		s.messages = s.messages[len(s.messages)-odomMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns a formatted string of the latest raw message in the buffer.
func (s *OdomSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", odomDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(odomDescriptor, latest.Message, ts)
	s.messages = nil

	return result
}

// Stop releases the underlying provider.
func (s *OdomSensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.stopped = true
	s.mu.Unlock()

	s.log.Info("UnitreeGo2Odom: stopping sensor")
	if s.provider != nil {
		s.provider.Stop()
	}
}
