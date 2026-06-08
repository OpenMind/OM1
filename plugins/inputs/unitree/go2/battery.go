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
	batteryprovider "github.com/openmind/om1/internal/providers/unitree/go2"
)

func init() {
	inputs.Register("UnitreeGo2Battery", NewUnitreeGo2Battery)
}

const (
	batteryDescriptor  = "Energy Levels"
	batteryPollPeriod  = 2 * time.Second
	batteryMaxMessages = 10

	batteryCriticalThreshold = 7.0
	batteryWarningThreshold  = 15.0

	batteryCriticalText = "CRITICAL: Your battery is almost empty. Immediately move to your charging station and recharge. If you cannot find your charging station, consider sitting down."
	batteryWarningText  = "WARNING: You are low on energy. Move to your charging station and recharge."
)

type BatteryConfig struct {
	Topic string `json:"topic"`
}

type BatterySensor struct {
	log      *zap.Logger
	provider *batteryprovider.BatteryZenohProvider

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

// NewUnitreeGo2Battery creates a new BatterySensor with the given configuration.
func NewUnitreeGo2Battery(configMap map[string]any) (inputs.Sensor, error) {
	var cfg BatteryConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	log := logger.Get().Named("UnitreeGo2Battery")
	log.Info("initializing", zap.String("topic", cfg.Topic))

	return &BatterySensor{
		log:      log,
		provider: batteryprovider.NewBatteryZenohProvider(cfg.Topic),
	}, nil
}

func (s *BatterySensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(batteryPollPeriod)
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

func (s *BatterySensor) Poll(_ context.Context) (any, error) {
	state := s.provider.State()
	return state, nil
}

func (s *BatterySensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	state, ok := raw.(batteryprovider.BatteryState)
	if !ok {
		return nil, nil
	}

	var res string
	switch {
	case state.Percentage < batteryCriticalThreshold:
		res = batteryCriticalText
	case state.Percentage < batteryWarningThreshold:
		res = batteryWarningText
	default:
		return nil, nil
	}

	msg := inputs.NewMessage(res)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > batteryMaxMessages {
		s.messages = s.messages[len(s.messages)-batteryMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

func (s *BatterySensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", batteryDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(batteryDescriptor, latest.Message, ts)
	s.messages = nil

	return result
}

func (s *BatterySensor) Stop() {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}

	s.stopped = true
	s.mu.Unlock()

	s.log.Info("stopping sensor")
	if s.provider != nil {
		s.provider.Stop()
	}
}
