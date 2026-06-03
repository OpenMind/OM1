package backgrounds

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

type fakeBackground struct {
	runs       atomic.Int32
	stops      atomic.Int32
	panicUntil int32
}

func (b *fakeBackground) Run(context.Context) {
	n := b.runs.Add(1)
	if n <= b.panicUntil {
		panic("boom")
	}
}
func (b *fakeBackground) Stop() { b.stops.Add(1) }

func TestRegisterAndLoad(t *testing.T) {
	want := &fakeBackground{}
	Register("FakeBG", func(map[string]any) (Background, error) { return want, nil })
	t.Cleanup(func() { delete(registry, "FakeBG") })

	got, err := Load("FakeBG", nil)
	require.NoError(t, err)
	require.Same(t, want, got)
}

func TestLoadUnknown(t *testing.T) {
	_, err := Load("missing", nil)
	require.Error(t, err)
	var unknown *UnknownPluginError
	require.ErrorAs(t, err, &unknown)
	require.Equal(t, "background plugin not found: missing", err.Error())
}

func TestOrchestratorRunsAndStops(t *testing.T) {
	bg := &fakeBackground{}
	o := NewOrchestrator([]Background{bg}, zap.NewNop())

	ctx, cancel := context.WithCancel(context.Background())
	done := o.Start(ctx)
	time.Sleep(20 * time.Millisecond)
	cancel()

	requireClosed(t, done, "Run is called repeatedly until cancel")
	require.Positive(t, bg.runs.Load(), "Run is called repeatedly")
	require.Equal(t, int32(1), bg.stops.Load(), "Stop is called once on shutdown")
}

func TestOrchestratorRecoversFromPanic(t *testing.T) {
	bg := &fakeBackground{panicUntil: 3}
	o := NewOrchestrator([]Background{bg}, zap.NewNop())

	ctx, cancel := context.WithCancel(context.Background())
	done := o.Start(ctx)
	time.Sleep(30 * time.Millisecond)
	cancel()

	requireClosed(t, done, "loop survives panics")
	require.Greater(t, bg.runs.Load(), int32(3), "loop keeps running after recovered panics")
}

func TestOrchestratorWaitsForAllBackgrounds(t *testing.T) {
	o := NewOrchestrator([]Background{&fakeBackground{}, &fakeBackground{}}, zap.NewNop())
	ctx, cancel := context.WithCancel(context.Background())
	done := o.Start(ctx)
	cancel()
	requireClosed(t, done, "done closes only after every background goroutine exits")
}

func requireClosed(t *testing.T, done <-chan struct{}, msg string) {
	t.Helper()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatalf("orchestrator did not finish after cancellation: %s", msg)
	}
}
