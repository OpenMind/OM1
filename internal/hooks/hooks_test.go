package hooks

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/providers/tts"
)

func TestFormatTemplate(t *testing.T) {
	out := formatTemplate("hello {name}, you are {age}", map[string]any{"name": "Ada", "age": 42})
	require.Equal(t, "hello Ada, you are 42", out)
	require.Equal(t, "no vars", formatTemplate("no vars", nil))
}

func TestStringVal(t *testing.T) {
	m := map[string]any{"s": "x", "n": 1}
	require.Equal(t, "x", stringVal(m, "s"))
	require.Equal(t, "", stringVal(m, "n"), "non-string returns empty")
	require.Equal(t, "", stringVal(m, "absent"))
}

func TestElevenLabsConfigFromDefaults(t *testing.T) {
	cfg := elevenLabsConfigFrom(map[string]any{})
	require.Equal(t, tts.DefaultVoiceID, cfg.VoiceID)
	require.Equal(t, tts.DefaultModelID, cfg.ModelID)
	require.Equal(t, tts.DefaultOutputFormat, cfg.OutputFormat)
	require.Equal(t, tts.DefaultRate, cfg.Rate)
}

func TestElevenLabsConfigFromOverrides(t *testing.T) {
	cfg := elevenLabsConfigFrom(map[string]any{
		"api_key":  "k",
		"voice_id": "v",
		"rate":     float64(3),
	})
	require.Equal(t, "k", cfg.APIKey)
	require.Equal(t, "v", cfg.VoiceID)
	require.Equal(t, 3, cfg.Rate)
}

func TestRunCommandHook(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "marker.txt")
	runner := NewHooks([]config.HookSpec{
		{HookType: "on_startup", HandlerType: "command", HandlerConfig: map[string]any{
			"command": "echo {msg} > " + marker,
		}},
		{HookType: "on_exit", HandlerType: "command", HandlerConfig: map[string]any{
			"command": "echo nope > " + marker + ".exit",
		}},
	}, nil, zap.NewNop())

	require.NoError(t, runner.Run(context.Background(), OnStartup, map[string]any{"msg": "started"}))

	data, err := os.ReadFile(marker)
	require.NoError(t, err)
	require.Contains(t, string(data), "started", "the matching command hook ran with substituted vars")
	require.NoFileExists(t, marker+".exit", "the on_exit hook did not run for an OnStartup trigger")
}

func TestRunSwallowsHookErrors(t *testing.T) {
	runner := NewHooks([]config.HookSpec{
		{HookType: "on_startup", HandlerType: "command", HandlerConfig: map[string]any{"command": "exit 1"}},
	}, nil, zap.NewNop())
	require.NoError(t, runner.Run(context.Background(), OnStartup, nil), "a failing hook is logged, not propagated")
}

func TestExecuteFunctionHook(t *testing.T) {
	called := false
	RegisterHook("testmod", "fn_"+t.Name(), func(_ *Runner, _ context.Context, _, vars map[string]any) error {
		called = true
		require.Equal(t, "v", vars["k"])
		return nil
	})

	runner := NewHooks([]config.HookSpec{
		{HookType: "on_entry", HandlerType: "function", HandlerConfig: map[string]any{
			"module_name": "testmod",
			"function":    "fn_" + t.Name(),
		}},
	}, nil, zap.NewNop())

	require.NoError(t, runner.Run(context.Background(), OnEntry, map[string]any{"k": "v"}))
	require.True(t, called, "the registered function hook was invoked")
}
