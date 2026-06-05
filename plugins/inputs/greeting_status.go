package inputs

import (
	"context"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("GreetingStatus", NewGreetingStatus)
}

type GreetingStatusSensor struct {
	log      *zap.Logger
	greeting *providers.GreetingConversationStateMachineProvider
}

// NewGreetingStatus constructs a GreetingStatusSensor.
func NewGreetingStatus(_ map[string]any) (inputs.Sensor, error) {
	log := logger.Get()
	log.Info("GreetingStatus: initializing")
	return &GreetingStatusSensor{
		log:      log,
		greeting: providers.Greeting(),
	}, nil
}

func (s *GreetingStatusSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		<-ctx.Done()
	}()
	return out, nil
}

func (s *GreetingStatusSensor) Poll(_ context.Context) (any, error) {
	return nil, nil
}

func (s *GreetingStatusSensor) RawToText(_ context.Context, _ any) (*inputs.Message, error) {
	return nil, nil
}

func (s *GreetingStatusSensor) FormattedLatestBuffer() string {
	return s.greeting.EndingGuidance()
}

func (s *GreetingStatusSensor) Stop() {
	s.log.Info("GreetingStatus: stopping sensor")
}
