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

// drainingSensor returns its text once, then an empty buffer, like real sensors that clear on read.
type drainingSensor struct {
	fakeSensor
	role Role
}

func (s *drainingSensor) Role() Role { return s.role }
func (s *drainingSensor) FormattedLatestBuffer() string {
	b := s.buffer
	s.buffer = ""
	return b
}

func TestObservationsKeepWorldStateAndDropConsumedEvents(t *testing.T) {
	cam := &drainingSensor{fakeSensor: fakeSensor{buffer: "camera: a desk"}, role: RoleWorld}
	voice := &drainingSensor{fakeSensor: fakeSensor{buffer: "hello robot"}, role: RoleConversation}
	alarm := &drainingSensor{fakeSensor: fakeSensor{buffer: "obstacle ahead"}, role: RoleEvent}
	o := NewOrchestrator([]Sensor{cam, voice, alarm}, zap.NewNop())

	first := o.Observations()
	require.Len(t, first, 3)
	require.Equal(t, Observation{Role: RoleWorld, Text: "camera: a desk", New: true, Age: first[0].Age}, first[0])
	require.True(t, first[0].New)
	require.Equal(t, RoleConversation, first[1].Role)
	require.Equal(t, RoleEvent, first[2].Role)

	second := o.Observations()
	require.Len(t, second, 1, "conversation and event text is consumed; world state persists")
	require.Equal(t, RoleWorld, second[0].Role)
	require.Equal(t, "camera: a desk", second[0].Text)
	require.False(t, second[0].New, "repeated world state is marked as not new")
	require.GreaterOrEqual(t, second[0].Age, time.Duration(0))

	cam.buffer = "camera: a person at the desk"
	third := o.Observations()
	require.Equal(t, "camera: a person at the desk", third[0].Text)
	require.True(t, third[0].New, "a fresh reading replaces the stored world state")
}

func TestObservationsEmptyWhenNothingArrived(t *testing.T) {
	o := NewOrchestrator([]Sensor{&drainingSensor{role: RoleWorld}, &drainingSensor{role: RoleEvent}}, zap.NewNop())
	require.Empty(t, o.Observations(), "a world sensor that never reported contributes nothing")
}
