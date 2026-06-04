package runtime

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/llm"
)

func TestCloneConfig(t *testing.T) {
	src := map[string]any{"a": 1}
	clone := cloneConfig(src)
	clone["a"] = 2
	require.Equal(t, 1, src["a"], "clone does not alias the source map")
	require.Equal(t, 2, clone["a"])
}

func TestAddMeta(t *testing.T) {
	cfg := addMeta(map[string]any{"api_key": "existing"}, map[string]string{"api_key": "new", "mode": "greeting"})
	require.Equal(t, "existing", cfg["api_key"], "existing keys are not overwritten")
	require.Equal(t, "greeting", cfg["mode"], "new meta keys are added")
}

func TestAddMetaNilConfig(t *testing.T) {
	cfg := addMeta(nil, map[string]string{"mode": "x"})
	require.Equal(t, "x", cfg["mode"], "a nil config map is allocated")
}

func TestMergePrompt(t *testing.T) {
	require.Equal(t, "mode", mergePrompt("global", "mode"), "mode-specific prompt wins")
	require.Equal(t, "global", mergePrompt("global", ""), "falls back to global when mode is empty")
}

func TestBuildMeta(t *testing.T) {
	m := &modeSetup{sys: &config.SystemConfig{APIKey: "k", RobotIP: "1.2.3.4"}}
	meta := m.buildMeta("greeting")
	require.Equal(t, map[string]string{
		"api_key":  "k",
		"robot_ip": "1.2.3.4",
		"mode":     "greeting",
	}, meta)
}

func TestBuildMetaOmitsEmpty(t *testing.T) {
	m := &modeSetup{sys: &config.SystemConfig{}}
	require.Empty(t, m.buildMeta(""), "no metadata when all source fields are empty")
}

func TestCollectSchemas(t *testing.T) {
	schema := map[string]any{"type": "function"}
	acts := []*actions.AgentAction{
		{LLMLabel: "a", Schema: schema},
		{LLMLabel: "b", Schema: schema, ExcludeFromPrompt: true},
		{LLMLabel: "c", Schema: nil},
	}
	got := collectSchemas(acts)
	require.Len(t, got, 1, "excluded and schema-less actions are dropped")
	require.Equal(t, schema, got[0], "the included schema is the one from action 'a'")
}

func TestToRuntimeConfig(t *testing.T) {
	m := &modeSetup{
		sys: &config.SystemConfig{Version: "v1", Name: "robot", SystemGovernance: "rules", UseTracer: true},
		cfg: config.ModeConfig{SystemPromptBase: "mode prompt", ActionExecMode: "sequential"},
	}
	rc := m.toRuntimeConfig()
	require.Equal(t, "v1", rc.Version)
	require.Equal(t, "mode prompt", rc.SystemPromptBase)
	require.Equal(t, "rules", rc.SystemGovernance)
	require.True(t, rc.UseTracer)
	require.Equal(t, 1.0, rc.Hertz, "zero hertz defaults to 1.0")
}

func TestToolCallsToMaps(t *testing.T) {
	got := toolCallsToMaps([]llm.ToolCall{{Name: "speak", Arguments: map[string]any{"text": "hi"}}})
	require.Len(t, got, 1)
	require.Equal(t, "speak", got[0]["name"])
	require.Equal(t, map[string]any{"text": "hi"}, got[0]["arguments"])
}

func TestTraceOutput(t *testing.T) {
	got := traceOutput(&llm.Response{ToolCalls: []llm.ToolCall{{Name: "move", Arguments: map[string]any{"dir": "left"}}}})
	require.Len(t, got, 1)
	require.Equal(t, "move", got[0]["type"])
	require.Equal(t, map[string]any{"dir": "left"}, got[0]["value"])
}

func TestLoadComponentsRequiresLLM(t *testing.T) {
	m := NewModeSetup(config.ModeConfig{Name: "greeting"}, &config.SystemConfig{})
	err := m.loadComponents()
	require.Error(t, err, "a mode with no LLM configured fails to load")
	require.Contains(t, err.Error(), "no LLM configured")
}

func TestLoadComponentsLoadsLLMAndMeta(t *testing.T) {
	var gotCfg map[string]any
	llm.Register("RuntimeFakeLLM", func(cfg map[string]any) (llm.LLM, error) {
		gotCfg = cfg
		return &fakeRuntimeLLM{}, nil
	})

	m := NewModeSetup(
		config.ModeConfig{Name: "greeting", CortexLLM: config.PluginSpec{Type: "RuntimeFakeLLM"}},
		&config.SystemConfig{APIKey: "secret"},
	)
	require.NoError(t, m.loadComponents())
	require.NotNil(t, m.cortexLLM)
	require.Equal(t, "secret", gotCfg["api_key"], "global api_key is injected as meta")
	require.Equal(t, "greeting", gotCfg["mode"])
}

// fakeRuntimeLLM is a no-op llm.LLM used to exercise component loading.
type fakeRuntimeLLM struct{ schemas []map[string]any }

func (f *fakeRuntimeLLM) Call(context.Context, string, []llm.Message) (*llm.Response, error) {
	return &llm.Response{}, nil
}
func (f *fakeRuntimeLLM) SetSchemas(s []map[string]any)     { f.schemas = s }
func (f *fakeRuntimeLLM) FunctionSchemas() []map[string]any { return f.schemas }
