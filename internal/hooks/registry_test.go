package hooks

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestRegisterAndLookupHook(t *testing.T) {
	fn := func(*Runner, context.Context, map[string]any, map[string]any) error { return nil }
	RegisterHook("mod", "lookup_"+t.Name(), fn)

	got, ok := lookupHook("mod", "lookup_"+t.Name())
	require.True(t, ok)
	require.NotNil(t, got)

	_, ok = lookupHook("mod", "never_registered")
	require.False(t, ok)
}

// withDefaultRunner installs a Runner for the duration of a test and restores
// whatever was there before.
func withDefaultRunner(t *testing.T, r *Runner) {
	t.Helper()

	defaultRunnerMu.RLock()
	prev := defaultRunner
	defaultRunnerMu.RUnlock()

	SetDefaultRunner(r)
	t.Cleanup(func() { SetDefaultRunner(prev) })
}

func TestInvokeRunsRegisteredHookWithTheModeRunner(t *testing.T) {
	var gotRunner *Runner
	var gotCfg map[string]any
	RegisterHook("mod", "invoke_"+t.Name(), func(r *Runner, _ context.Context, cfg, _ map[string]any) error {
		gotRunner, gotCfg = r, cfg
		return nil
	})

	// The runtime publishes the active mode's Runner so a manual trigger
	// inherits its memory manager rather than losing memory recall.
	runner := NewHooks(nil, nil, zap.NewNop())
	withDefaultRunner(t, runner)

	cfg := map[string]any{"robot_name": "Iris"}
	require.NoError(t, Invoke(context.Background(), "mod", "invoke_"+t.Name(), cfg, nil))
	require.Same(t, runner, gotRunner)
	require.Equal(t, cfg, gotCfg)
}

func TestInvokePropagatesHookError(t *testing.T) {
	sentinel := errors.New("boom")
	RegisterHook("mod", "err_"+t.Name(), func(*Runner, context.Context, map[string]any, map[string]any) error {
		return sentinel
	})

	withDefaultRunner(t, NewHooks(nil, nil, zap.NewNop()))

	require.ErrorIs(t, Invoke(context.Background(), "mod", "err_"+t.Name(), nil, nil), sentinel)
}

func TestInvokeWithoutDefaultRunner(t *testing.T) {
	RegisterHook("mod", "norunner_"+t.Name(), func(*Runner, context.Context, map[string]any, map[string]any) error {
		t.Fatal("hook must not run without a Runner to supply its logger")
		return nil
	})

	withDefaultRunner(t, nil)

	require.ErrorContains(t,
		Invoke(context.Background(), "mod", "norunner_"+t.Name(), nil, nil), "no default runner")
}

func TestInvokeUnknownHook(t *testing.T) {
	withDefaultRunner(t, NewHooks(nil, nil, zap.NewNop()))

	require.ErrorContains(t,
		Invoke(context.Background(), "mod", "never_registered", nil, nil), "unknown function hook")
}

func TestRegisterHookDuplicatePanics(t *testing.T) {
	fn := func(*Runner, context.Context, map[string]any, map[string]any) error { return nil }
	RegisterHook("dup", "fn_"+t.Name(), fn)
	require.PanicsWithValue(t,
		"hooks: duplicate function hook registration for dup.fn_"+t.Name(),
		func() { RegisterHook("dup", "fn_"+t.Name(), fn) },
	)
}
