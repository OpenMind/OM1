package inputs

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

type channelSensor struct {
	fakeSensor
	readings []any
	stopped  chan struct{}
}

func newChannelSensor(triggers bool, readings ...any) *channelSensor {
	return &channelSensor{
		fakeSensor: fakeSensor{triggers: triggers},
		readings:   readings,
		stopped:    make(chan struct{}, 1),
	}
}

func (s *channelSensor) Listen(ctx context.Context) (<-chan any, error) {
	ch := make(chan any)
	go func() {
		defer close(ch)
		for _, r := range s.readings {
			select {
			case ch <- r:
			case <-ctx.Done():
				return
			}
		}
		<-ctx.Done()
	}()
	return ch, nil
}

func (s *channelSensor) RawToText(_ context.Context, raw any) (*Message, error) {
	return NewMessage(raw.(string)), nil
}
func (s *channelSensor) TriggersTick() bool { return s.triggers }
func (s *channelSensor) Stop()              { s.stopped <- struct{}{} }

func TestBuffersSnapshot(t *testing.T) {
	o := NewOrchestrator([]Sensor{
		&fakeSensor{buffer: "one"},
		&fakeSensor{buffer: "two"},
	}, zap.NewNop())

	require.Equal(t, []string{"one", "two"}, o.Buffers())
}

func TestTickNowFiresForTriggeringSensor(t *testing.T) {
	o := NewOrchestrator([]Sensor{newChannelSensor(true, "hi")}, zap.NewNop())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	o.Start(ctx)

	select {
	case <-o.TickNow():
	case <-time.After(time.Second):
		t.Fatal("expected a tick signal from a triggering sensor")
	}
}

func TestTickNowSilentForNonTriggeringSensor(t *testing.T) {
	o := NewOrchestrator([]Sensor{newChannelSensor(false, "hi")}, zap.NewNop())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	o.Start(ctx)

	select {
	case <-o.TickNow():
		t.Fatal("non-triggering sensor must not wake the loop")
	case <-time.After(100 * time.Millisecond):
	}
}

func TestStartStopsSensorsOnCancel(t *testing.T) {
	sensor := newChannelSensor(false)
	o := NewOrchestrator([]Sensor{sensor}, zap.NewNop())

	ctx, cancel := context.WithCancel(context.Background())
	done := o.Start(ctx)
	cancel()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Start did not return after cancellation")
	}
	select {
	case <-sensor.stopped:
	case <-time.After(time.Second):
		t.Fatal("sensor.Stop was not called on cancellation")
	}
}
