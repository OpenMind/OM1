package providers

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func newIO() *IOProvider { return &IOProvider{inputs: make(map[string]Input)} }

func TestIOSingleton(t *testing.T) {
	require.Same(t, IO(), IO(), "IO returns the same singleton")
}

func TestAddAndGetInput(t *testing.T) {
	p := newIO()
	ts := time.Unix(1000, 0)
	p.AddInput("Voice", "hello", ts)

	got := p.GetInput("Voice")
	require.NotNil(t, got)
	require.Equal(t, "hello", got.Input)
	require.Equal(t, ts, got.Timestamp)
	require.Equal(t, 0, got.Tick, "tick recorded at insertion time")
}

func TestGetInputReturnsCopy(t *testing.T) {
	p := newIO()
	p.AddInput("Voice", "hello", time.Unix(1, 0))
	got := p.GetInput("Voice")
	got.Input = "mutated"
	require.Equal(t, "hello", p.GetInput("Voice").Input, "callers receive a copy, not the stored value")
}

func TestGetInputMissing(t *testing.T) {
	require.Nil(t, newIO().GetInput("absent"))
}

func TestAddInputDefaultsTimestamp(t *testing.T) {
	p := newIO()
	before := time.Now()
	p.AddInput("Voice", "hi", time.Time{})
	got := p.GetInput("Voice")
	require.False(t, got.Timestamp.IsZero(), "zero timestamp is replaced with now")
	require.False(t, got.Timestamp.Before(before))
}

func TestRemoveInput(t *testing.T) {
	p := newIO()
	p.AddInput("Voice", "hi", time.Now())
	p.RemoveInput("Voice")
	require.Nil(t, p.GetInput("Voice"))
	require.NotPanics(t, func() { p.RemoveInput("absent") })
}

func TestInputTimestamp(t *testing.T) {
	p := newIO()
	p.AddInput("Voice", "hi", time.Unix(1, 0))

	newTS := time.Unix(2, 0)
	p.AddInputTimestamp("Voice", newTS)
	ts, ok := p.GetInputTimestamp("Voice")
	require.True(t, ok)
	require.Equal(t, newTS, ts)
	require.Equal(t, "hi", p.GetInput("Voice").Input, "value and tick are preserved")

	_, ok = p.GetInputTimestamp("absent")
	require.False(t, ok)
	require.NotPanics(t, func() { p.AddInputTimestamp("absent", newTS) }, "no-op for missing key")
}

func TestTickCounter(t *testing.T) {
	p := newIO()
	require.Equal(t, 0, p.TickCounter())
	require.Equal(t, 1, p.IncrementTick())
	require.Equal(t, 2, p.IncrementTick())
	require.Equal(t, 2, p.TickCounter())
	p.ResetTickCounter()
	require.Equal(t, 0, p.TickCounter())
}

func TestRecordTickAndTotals(t *testing.T) {
	p := newIO()
	require.Equal(t, int64(0), p.TotalTicks())
	p.RecordTick(time.Now())
	p.RecordTick(time.Now())
	require.Equal(t, int64(2), p.TotalTicks())
}
