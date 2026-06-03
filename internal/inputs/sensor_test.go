package inputs

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

type fakeSensor struct {
	buffer   string
	triggers bool
}

func (s *fakeSensor) Listen(context.Context) (<-chan any, error)       { return nil, nil }
func (s *fakeSensor) Poll(context.Context) (any, error)                { return nil, nil }
func (s *fakeSensor) RawToText(context.Context, any) (*Message, error) { return nil, nil }
func (s *fakeSensor) FormattedLatestBuffer() string                    { return s.buffer }
func (s *fakeSensor) Stop()                                            {}

type triggeringSensor struct {
	fakeSensor
	wakes bool
}

func (s *triggeringSensor) TriggersTick() bool { return s.wakes }

func TestNewMessage(t *testing.T) {
	msg := NewMessage("hello")
	require.Equal(t, "hello", msg.Message)
	require.Positive(t, msg.Timestamp, "timestamp is set to a positive epoch value")
}

func TestRegisterAndLoad(t *testing.T) {
	want := &fakeSensor{}
	Register("FakeSensor", func(map[string]any) (Sensor, error) { return want, nil })
	t.Cleanup(func() { delete(registry, "FakeSensor") })

	got, err := Load("FakeSensor", nil)
	require.NoError(t, err)
	require.Same(t, want, got)
}

func TestLoadUnknown(t *testing.T) {
	_, err := Load("missing", nil)
	require.Error(t, err)
	var unknown *UnknownPluginError
	require.ErrorAs(t, err, &unknown)
	require.Equal(t, "input plugin not found: missing", err.Error())
}

func TestTriggersTick(t *testing.T) {
	require.False(t, triggersTick(&fakeSensor{}), "sensor without TickTrigger never wakes the loop")
	require.False(t, triggersTick(&triggeringSensor{wakes: false}))
	require.True(t, triggersTick(&triggeringSensor{wakes: true}))
}
