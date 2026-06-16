package runtime

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/llm"
)

type countingBackground struct {
	runs  atomic.Int32
	stops atomic.Int32
}

func (b *countingBackground) Run(ctx context.Context) {
	b.runs.Add(1)

	select {
	case <-ctx.Done():
	case <-time.After(time.Millisecond):
	}
}

func (b *countingBackground) Stop() { b.stops.Add(1) }

func TestGlobalBackgroundsSurviveModeContextCancel(t *testing.T) {
	bg := &countingBackground{}
	rt := &Runtime{
		log: zap.NewNop(),
		globalBg: globalBackgroundState{
			orchestrator: backgrounds.NewOrchestrator([]backgrounds.Background{bg}, zap.NewNop()),
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rt.startGlobalBackgrounds(ctx)
	require.NotNil(t, rt.globalBg.done, "starting populates the done channel")
	require.NotNil(t, rt.globalBg.cancel, "starting populates the cancel func")

	_, modeCancel := context.WithCancel(ctx)
	modeCancel()

	require.Eventually(t, func() bool { return bg.runs.Load() > 1 }, time.Second, 5*time.Millisecond,
		"global background keeps running across a mode-context cancellation")
	require.Zero(t, bg.stops.Load(), "global background is not stopped by a mode switch")

	rt.stopGlobalBackgrounds()
	require.Equal(t, int32(1), bg.stops.Load(), "global background stops exactly once on shutdown")
}

func TestStartGlobalBackgroundsNoOpWhenNotConfigured(t *testing.T) {
	rt := &Runtime{log: zap.NewNop()}
	rt.startGlobalBackgrounds(context.Background())
	require.Nil(t, rt.globalBg.done)
	require.Nil(t, rt.globalBg.cancel)
}

func TestStartGlobalBackgroundsNoOpWhenContextCancelled(t *testing.T) {
	rt := &Runtime{
		log: zap.NewNop(),
		globalBg: globalBackgroundState{
			orchestrator: backgrounds.NewOrchestrator([]backgrounds.Background{&countingBackground{}}, zap.NewNop()),
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rt.startGlobalBackgrounds(ctx)
	require.Nil(t, rt.globalBg.done, "an already-cancelled context skips startup")
}

func TestStopGlobalBackgroundsNoOpWhenNotStarted(t *testing.T) {
	rt := &Runtime{log: zap.NewNop()}
	require.NotPanics(t, rt.stopGlobalBackgrounds, "stopping before start is a no-op")
}

type signalLLM struct {
	called  chan struct{}
	once    sync.Once
	schemas []map[string]any
}

func (l *signalLLM) Call(context.Context, string, []llm.Message) (*llm.Response, error) {
	l.once.Do(func() { close(l.called) })
	return &llm.Response{}, nil
}
func (l *signalLLM) SetSchemas(s []map[string]any)     { l.schemas = s }
func (l *signalLLM) FunctionSchemas() []map[string]any { return l.schemas }

func soloConfig(llmType string) *config.SystemConfig {
	return &config.SystemConfig{
		Name:        "robot",
		DefaultMode: "solo",
		Modes: map[string]config.ModeConfig{
			"solo": {
				Name:      "solo",
				Hertz:     50,
				CortexLLM: config.PluginSpec{Type: llmType},
			},
		},
	}
}

func TestNew(t *testing.T) {
	sys := soloConfig("whatever")
	rt := New(sys, zap.NewNop(), Options{})
	require.Same(t, sys, rt.systemConfig)
	require.NotNil(t, rt.manager, "a mode manager is created")
	require.NotNil(t, rt.ioProvider)
	require.NotNil(t, rt.tracer)
	require.NotNil(t, rt.modeTransitionCh)
	require.Equal(t, "solo", rt.manager.CurrentMode())
}

func TestScheduleTransition(t *testing.T) {
	rt := &Runtime{log: zap.NewNop(), modeTransitionCh: make(chan string, 1)}

	require.False(t, rt.scheduleTransition(""), "an empty target schedules nothing")

	require.True(t, rt.scheduleTransition("active"), "a non-empty target is scheduled")
	require.True(t, rt.scheduleTransition("idle"), "a second target returns true but is dropped (channel full)")

	select {
	case got := <-rt.modeTransitionCh:
		require.Equal(t, "active", got, "the first scheduled target is kept; later ones are dropped while one is pending")
	default:
		t.Fatal("expected a pending transition")
	}
}

func TestInitializeModeUnknownMode(t *testing.T) {
	rt := New(soloConfig("whatever"), zap.NewNop(), Options{})
	err := rt.initializeMode("ghost")
	require.Error(t, err)
	require.Contains(t, err.Error(), "not found")
}

func TestInitializeModeSuccess(t *testing.T) {
	llm.Register("InitFakeLLM", func(map[string]any) (llm.LLM, error) { return &fakeRuntimeLLM{}, nil })
	backgrounds.Register("InitFakeBG", func(map[string]any) (backgrounds.Background, error) { return &countingBackground{}, nil })

	sys := soloConfig("InitFakeLLM")
	mode := sys.Modes["solo"]
	mode.AgentBackgrounds = []config.PluginSpec{{Type: "InitFakeBG"}}
	sys.Modes["solo"] = mode

	rt := New(sys, zap.NewNop(), Options{})
	require.NoError(t, rt.initializeMode("solo"))

	require.NotNil(t, rt.current, "the mode state is populated")
	require.NotNil(t, rt.current.cortexLLM)
	require.NotNil(t, rt.current.promptFuser)
	require.NotNil(t, rt.current.bgOrchestrator, "the mode's background orchestrator is created")
}

func TestStartStopOrchestrators(t *testing.T) {
	signal := &signalLLM{called: make(chan struct{})}
	llm.Register("SignalLLM", func(map[string]any) (llm.LLM, error) { return signal, nil })

	rt := New(soloConfig("SignalLLM"), zap.NewNop(), Options{})
	require.NoError(t, rt.initializeMode("solo"))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rt.startOrchestrators(ctx)

	select {
	case <-signal.called:
	case <-time.After(2 * time.Second):
		t.Fatal("cortex loop did not tick within the timeout")
	}

	rt.stopOrchestrators()
	require.Nil(t, rt.current, "stopOrchestrators clears the current mode state")
}

// TestRunStartsAndStops exercises the full Run loop: it starts the global
// background, ticks the cortex loop, and on context cancellation shuts both down
// cleanly, returning the context error.
func TestRunStartsAndStops(t *testing.T) {
	signal := &signalLLM{called: make(chan struct{})}
	llm.Register("RunLLM", func(map[string]any) (llm.LLM, error) { return signal, nil })
	bg := &countingBackground{}
	backgrounds.Register("RunBG", func(map[string]any) (backgrounds.Background, error) { return bg, nil })

	sys := soloConfig("RunLLM")
	sys.GlobalBackgrounds = []config.PluginSpec{{Type: "RunBG"}}

	rt := New(sys, zap.NewNop(), Options{})

	ctx, cancel := context.WithCancel(context.Background())
	runErr := make(chan error, 1)
	go func() { runErr <- rt.Run(ctx) }()

	select {
	case <-signal.called:
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatal("cortex loop did not tick within the timeout")
	}

	cancel()
	select {
	case err := <-runErr:
		require.ErrorIs(t, err, context.Canceled, "Run returns the cancellation error")
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not return after cancellation")
	}

	require.Positive(t, bg.runs.Load(), "the global background ran")
	require.Equal(t, int32(1), bg.stops.Load(), "the global background is stopped on shutdown")
}

func TestOnModeTransition(t *testing.T) {
	llm.Register("TransLLM", func(map[string]any) (llm.LLM, error) { return &fakeRuntimeLLM{}, nil })

	sys := soloConfig("TransLLM")
	sys.Modes["other"] = config.ModeConfig{Name: "other", Hertz: 50, CortexLLM: config.PluginSpec{Type: "TransLLM"}}

	rt := New(sys, zap.NewNop(), Options{})
	require.NoError(t, rt.initializeMode("solo"))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	rt.startOrchestrators(ctx)

	require.NoError(t, rt.onModeTransition(ctx, "solo", "other"))
	require.Equal(t, "other", rt.manager.CurrentMode(), "the manager reflects the new mode")
	require.NotNil(t, rt.current, "the new mode is initialised and running")

	rt.stopOrchestrators()
}

func TestWatchConfigReturnsEarly(t *testing.T) {
	ctx := context.Background()

	noName := &Runtime{log: zap.NewNop(), systemConfig: &config.SystemConfig{}}
	require.NotPanics(t, func() { noName.watchConfig(ctx) }, "an empty name returns immediately")

	missingFile := &Runtime{log: zap.NewNop(), systemConfig: &config.SystemConfig{Name: "definitely-missing-config-xyz"}}
	require.NotPanics(t, func() { missingFile.watchConfig(ctx) }, "a missing config file returns immediately")
}
