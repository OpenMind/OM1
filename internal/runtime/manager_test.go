package runtime

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
)

func twoModeConfig(rules ...config.TransitionRule) *config.SystemConfig {
	return &config.SystemConfig{
		DefaultMode: "idle",
		Modes: map[string]config.ModeConfig{
			"idle":   {Name: "idle"},
			"active": {Name: "active"},
		},
		TransitionRules: rules,
	}
}

func newTestManager(t *testing.T, sys *config.SystemConfig) *ModeManager {
	t.Helper()
	return NewModeManager(sys, zap.NewNop())
}

func TestJoinStrings(t *testing.T) {
	require.Equal(t, "", joinStrings(nil))
	require.Equal(t, " a b", joinStrings([]string{"a", "b"}))
}

func TestHighestPriorityTarget(t *testing.T) {
	require.Equal(t, "", highestPriorityTarget(nil), "no rules yields no target")

	rules := []config.TransitionRule{
		{ToMode: "low", Priority: 1},
		{ToMode: "high", Priority: 5},
		{ToMode: "mid", Priority: 3},
	}
	require.Equal(t, "high", highestPriorityTarget(rules), "the highest-priority rule wins")

	tie := []config.TransitionRule{
		{ToMode: "first", Priority: 2},
		{ToMode: "second", Priority: 2},
	}
	require.Equal(t, "first", highestPriorityTarget(tie), "ties resolve to the first rule (stable)")
}

func TestEvaluateSingleCondition(t *testing.T) {
	tests := []struct {
		name     string
		expected any
		ctx      map[string]any
		want     bool
	}{
		{"missing key", "x", map[string]any{}, false},
		{"string equal", "x", map[string]any{"k": "x"}, true},
		{"string mismatch", "x", map[string]any{"k": "y"}, false},
		{"bool equal", true, map[string]any{"k": true}, true},
		{"min/max in range", map[string]any{"min": 5.0, "max": 10.0}, map[string]any{"k": 7}, true},
		{"min/max below", map[string]any{"min": 5.0}, map[string]any{"k": 4}, false},
		{"min/max above", map[string]any{"max": 10.0}, map[string]any{"k": 11}, false},
		{"min/max non-numeric actual", map[string]any{"min": 5.0}, map[string]any{"k": "nope"}, false},
		{"contains match (case-insensitive)", map[string]any{"contains": "ELL"}, map[string]any{"k": "Hello"}, true},
		{"contains miss", map[string]any{"contains": "zzz"}, map[string]any{"k": "Hello"}, false},
		{"contains non-string actual", map[string]any{"contains": "1"}, map[string]any{"k": 1}, false},
		{"one_of match", map[string]any{"one_of": []any{"a", "b"}}, map[string]any{"k": "b"}, true},
		{"one_of miss", map[string]any{"one_of": []any{"a", "b"}}, map[string]any{"k": "c"}, false},
		{"not satisfied", map[string]any{"not": "x"}, map[string]any{"k": "y"}, true},
		{"not violated", map[string]any{"not": "x"}, map[string]any{"k": "x"}, false},
		{"unknown operator map", map[string]any{"bogus": 1}, map[string]any{"k": "x"}, false},
		{"slice membership match", []any{"a", "b"}, map[string]any{"k": "a"}, true},
		{"slice membership miss", []any{"a", "b"}, map[string]any{"k": "c"}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, evaluateSingleCondition("k", tc.expected, tc.ctx))
		})
	}
}

func TestEvaluateContextConditions(t *testing.T) {
	require.True(t, evaluateContextConditions(map[string]any{}, map[string]any{}), "empty conditions always match")
	conds := map[string]any{"a": true, "b": "x"}
	require.True(t, evaluateContextConditions(conds, map[string]any{"a": true, "b": "x"}), "all conditions match")
	require.False(t, evaluateContextConditions(conds, map[string]any{"a": true, "b": "y"}), "one failing condition fails the set")
}

func TestCooldownExpired(t *testing.T) {
	m := newTestManager(t, twoModeConfig())

	require.True(t, m.cooldownExpired(config.TransitionRule{CooldownSeconds: 0}), "no cooldown configured is always expired")

	rule := config.TransitionRule{FromMode: "idle", ToMode: "active", CooldownSeconds: 10}
	require.True(t, m.cooldownExpired(rule), "no recorded transition means expired")

	m.cooldowns["idle→active"] = time.Now()
	require.False(t, m.cooldownExpired(rule), "a recent transition is still cooling down")

	m.cooldowns["idle→active"] = time.Now().Add(-time.Minute)
	require.True(t, m.cooldownExpired(rule), "an old transition has cooled down")
}

func TestNewModeManagerDefaults(t *testing.T) {
	m := newTestManager(t, twoModeConfig())
	require.Equal(t, "idle", m.CurrentMode(), "starts in the configured default mode")
	require.NotZero(t, m.state.ModeStartTime, "mode start time is initialised")
}

func TestUpdateAndResetUserContext(t *testing.T) {
	m := newTestManager(t, twoModeConfig())
	m.UpdateUserContext(map[string]any{"a": 1})
	m.UpdateUserContext(map[string]any{"b": 2})
	require.Equal(t, map[string]any{"a": 1, "b": 2}, m.userContext, "updates merge into the context")

	m.ResetUserContext()
	require.Empty(t, m.userContext, "reset clears the context")
}

func TestClose(t *testing.T) {
	m := newTestManager(t, twoModeConfig())
	require.NotPanics(t, m.Close)
}

func TestCheckTransitionsNoneMatch(t *testing.T) {
	m := newTestManager(t, twoModeConfig())
	require.Equal(t, "", m.CheckTransitions(context.Background(), []string{"nothing relevant"}))
}

func TestCheckTransitionsInputTriggered(t *testing.T) {
	m := newTestManager(t, twoModeConfig(config.TransitionRule{
		FromMode: "idle", ToMode: "active", TransitionType: "input_triggered",
		TriggerKeywords: []string{"wake"}, Priority: 1,
	}))
	require.Equal(t, "active", m.CheckTransitions(context.Background(), []string{"please WAKE up"}),
		"a matching keyword (case-insensitive) triggers the transition")
	require.Equal(t, "", m.CheckTransitions(context.Background(), []string{"stay asleep"}),
		"no keyword match yields no transition")
}

func TestCheckTransitionsInputTriggeredWildcardAndPriority(t *testing.T) {
	m := newTestManager(t, twoModeConfig(
		config.TransitionRule{FromMode: "*", ToMode: "idle", TransitionType: "input_triggered", TriggerKeywords: []string{"go"}, Priority: 1},
		config.TransitionRule{FromMode: "idle", ToMode: "active", TransitionType: "input_triggered", TriggerKeywords: []string{"go"}, Priority: 5},
	))
	require.Equal(t, "active", m.CheckTransitions(context.Background(), []string{"go"}),
		"the wildcard rule matches but the higher-priority specific rule wins")
}

func TestCheckTransitionsUnknownTargetSkipped(t *testing.T) {
	m := newTestManager(t, twoModeConfig(config.TransitionRule{
		FromMode: "idle", ToMode: "ghost", TransitionType: "input_triggered", TriggerKeywords: []string{"wake"},
	}))
	require.Equal(t, "", m.CheckTransitions(context.Background(), []string{"wake"}),
		"a rule pointing at an undefined mode is ignored")
}

func TestCheckTransitionsContextAware(t *testing.T) {
	m := newTestManager(t, twoModeConfig(config.TransitionRule{
		FromMode: "idle", ToMode: "active", TransitionType: "context_aware",
		ContextConditions: map[string]any{"awake": true},
	}))
	require.Equal(t, "", m.CheckTransitions(context.Background(), nil), "no matching context yet")

	m.UpdateUserContext(map[string]any{"awake": true})
	require.Equal(t, "active", m.CheckTransitions(context.Background(), nil),
		"the published context satisfies the rule")
}

func TestCheckTransitionsTimeBased(t *testing.T) {
	m := newTestManager(t, twoModeConfig(config.TransitionRule{
		FromMode: "idle", ToMode: "active", TransitionType: "time_based", TimeoutSeconds: 1,
	}))
	require.Equal(t, "", m.CheckTransitions(context.Background(), nil), "not enough time has elapsed")

	m.state.ModeStartTime = time.Now().Add(-2 * time.Second)
	require.Equal(t, "active", m.CheckTransitions(context.Background(), nil),
		"the mode has been active past the timeout")
}

func TestCheckTransitionsRespectsCooldown(t *testing.T) {
	m := newTestManager(t, twoModeConfig(config.TransitionRule{
		FromMode: "idle", ToMode: "active", TransitionType: "input_triggered",
		TriggerKeywords: []string{"wake"}, CooldownSeconds: 60,
	}))
	m.cooldowns["idle→active"] = time.Now()
	require.Equal(t, "", m.CheckTransitions(context.Background(), []string{"wake"}),
		"a rule still within its cooldown does not fire")
}

func TestTransitionUpdatesState(t *testing.T) {
	m := newTestManager(t, twoModeConfig())

	m.Transition("active", "test", nil, nil)

	require.Equal(t, "active", m.CurrentMode())
	require.Equal(t, "idle", m.state.PreviousMode)
	require.Equal(t, []string{"idle"}, m.state.TransitionHistory)
	require.Contains(t, m.cooldowns, "idle→active", "the transition records a cooldown timestamp")
}

func TestTransitionHistoryCapped(t *testing.T) {
	m := newTestManager(t, twoModeConfig())
	for i := 0; i < 25; i++ {
		if i%2 == 0 {
			m.Transition("active", "test", nil, nil)
		} else {
			m.Transition("idle", "test", nil, nil)
		}
	}
	require.Len(t, m.state.TransitionHistory, 20, "history is capped at 20 entries")
}

func TestSaveAndLoadRoundTrip(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "mode_state.json")

	sys := twoModeConfig()
	sys.ModeMemoryEnabled = true

	writer := newTestManager(t, sys)
	writer.statePath = statePath
	writer.Transition("active", "test", nil, nil)

	reader := newTestManager(t, sys)
	reader.statePath = statePath
	reader.load()
	require.Equal(t, "active", reader.CurrentMode(), "saved mode is restored on load")
}

func TestLoadIgnoresUnknownMode(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "mode_state.json")
	require.NoError(t, os.WriteFile(statePath, []byte(`{"current_mode":"ghost"}`), 0o644))

	m := newTestManager(t, twoModeConfig())
	m.statePath = statePath
	m.load()
	require.Equal(t, "idle", m.CurrentMode(), "a saved mode not present in config is ignored")
}
