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
	localizationprovider "github.com/openmind/om1/internal/providers/unitree/go2"
)

func init() {
	inputs.Register("LocalizationInput", NewLocalization)
}

const (
	localizationDescriptor  = "Robot localization status - indicates if navigation is safe to proceed."
	localizationPollPeriod  = 100 * time.Millisecond
	localizationMaxMessages = 10

	localizedText    = "LOCALIZED: Robot position is confirmed. Navigation commands are safe to execute."
	notLocalizedText = "NOT LOCALIZED: Robot position uncertain. DO NOT attempt navigation until localized."
)

type LocalizationConfig struct {
	Topic            string  `json:"topic"`
	QualityTolerance float32 `json:"quality_tolerance"`
}

type LocalizationSensor struct {
	log      *zap.Logger
	provider *localizationprovider.LocalizationProvider

	mu       sync.Mutex
	messages []inputs.Message
	stopped  bool
}

// NewLocalization creates a new LocalizationSensor from the config map.
func NewLocalization(configMap map[string]any) (inputs.Sensor, error) {
	var cfg LocalizationConfig
	if b, err := json.Marshal(configMap); err == nil {
		_ = json.Unmarshal(b, &cfg)
	}

	log := logger.Get().Named("LocalizationInput")
	log.Info("initializing",
		zap.String("topic", cfg.Topic),
		zap.Float32("quality_tolerance", cfg.QualityTolerance),
	)

	return &LocalizationSensor{
		log:      log,
		provider: localizationprovider.NewLocalizationProvider(cfg.Topic, cfg.QualityTolerance),
	}, nil
}

// Listen starts a goroutine that polls the provider at a regular interval.
func (s *LocalizationSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		defer s.Stop()

		ticker := time.NewTicker(localizationPollPeriod)
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

// Poll returns the latest localization status text from the provider.
func (s *LocalizationSensor) Poll(_ context.Context) (any, error) {
	if s.provider.IsLocalized() && s.provider.Pose() != nil {
		return localizedText, nil
	}
	return notLocalizedText, nil
}

func (s *LocalizationSensor) RawToText(_ context.Context, raw any) (*inputs.Message, error) {
	text, ok := raw.(string)
	if !ok || text == "" {
		return nil, nil
	}

	msg := inputs.NewMessage(text)

	s.mu.Lock()
	s.messages = append(s.messages, *msg)
	if len(s.messages) > localizationMaxMessages {
		s.messages = s.messages[len(s.messages)-localizationMaxMessages:]
	}
	s.mu.Unlock()

	return msg, nil
}

// FormattedLatestBuffer returns the most recent status message in the buffer.
func (s *LocalizationSensor) FormattedLatestBuffer() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.messages) == 0 {
		return ""
	}

	latest := s.messages[len(s.messages)-1]
	result := fmt.Sprintf("\n%s: '%s'\n", localizationDescriptor, latest.Message)

	ts := time.Unix(0, int64(latest.Timestamp*1e9))
	providers.IO().AddInput(localizationDescriptor, latest.Message, ts)
	s.messages = nil

	return result
}

// Stop releases the underlying provider.
func (s *LocalizationSensor) Stop() {
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
