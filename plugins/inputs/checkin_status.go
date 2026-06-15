package inputs

import (
	"context"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/logger"
	"github.com/openmind/om1/internal/providers"
)

func init() {
	inputs.Register("CheckinStatus", NewCheckinStatus)
}

type CheckinStatusSensor struct {
	log *zap.Logger
}

func NewCheckinStatus(_ map[string]any) (inputs.Sensor, error) {
	log := logger.Get().Named("CheckinStatus")
	log.Info("initializing")
	return &CheckinStatusSensor{log: log}, nil
}

func (s *CheckinStatusSensor) Listen(ctx context.Context) (<-chan any, error) {
	out := make(chan any)
	go func() {
		defer close(out)
		<-ctx.Done()
	}()
	return out, nil
}

func (s *CheckinStatusSensor) Poll(_ context.Context) (any, error) {
	return nil, nil
}

func (s *CheckinStatusSensor) RawToText(_ context.Context, _ any) (*inputs.Message, error) {
	return nil, nil
}

func (s *CheckinStatusSensor) FormattedLatestBuffer() string {
	in := providers.IO().GetInput("CheckinStatus")
	if in == nil || in.Input == "" {
		return ""
	}
	return "\n" + in.Input + "\n"
}

func (s *CheckinStatusSensor) TriggersTick() bool {
	in := providers.IO().GetInput("CheckinStatus")
	return in != nil && in.Input != ""
}
func (s *CheckinStatusSensor) Stop()              {}
