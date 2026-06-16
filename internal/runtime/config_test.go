package runtime

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/backgrounds"
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
	cfg := addMeta(map[string]any{"api_key": "existing"}, map[string]any{"api_key": "new", "mode": "greeting"})
	require.Equal(t, "existing", cfg["api_key"], "existing keys are not overwritten")
	require.Equal(t, "greeting", cfg["mode"], "new meta keys are added")
}

func TestAddMetaNilConfig(t *testing.T) {
	cfg := addMeta(nil, map[string]any{"mode": "x"})
	require.Equal(t, "x", cfg["mode"], "a nil config map is allocated")
}

func TestMergePrompt(t *testing.T) {
	require.Equal(t, "mode", mergePrompt("global", "mode"), "mode-specific prompt wins")
	require.Equal(t, "global", mergePrompt("global", ""), "falls back to global when mode is empty")
}

func TestBuildMeta(t *testing.T) {
	m := &modeSetup{sys: &config.SystemConfig{APIKey: "k", RobotIP: "1.2.3.4", UseSim: true}}
	meta := m.buildMeta("greeting")
	require.Equal(t, map[string]any{
		"api_key":  "k",
		"robot_ip": "1.2.3.4",
		"use_sim":  true,
		"mode":     "greeting",
	}, meta)
}

func TestBuildMetaOmitsUseSimWhenFalse(t *testing.T) {
	m := &modeSetup{sys: &config.SystemConfig{APIKey: "k"}}
	_, ok := m.buildMeta("greeting")["use_sim"]
	require.False(t, ok, "use_sim is omitted when not enabled, like other zero-value fields")
}

func TestBuildMetaOmitsEmpty(t *testing.T) {
	m := &modeSetup{sys: &config.SystemConfig{}}
	require.Empty(t, m.buildMeta(""), "no metadata when all source fields are empty")
}

func TestBuildSystemMeta(t *testing.T) {
	meta := buildSystemMeta(&config.SystemConfig{APIKey: "k", RobotIP: "1.2.3.4", URID: "u1", UseSim: true})
	require.Equal(t, map[string]any{
		"api_key":  "k",
		"robot_ip": "1.2.3.4",
		"URID":     "u1",
		"use_sim":  true,
	}, meta)
	_, hasMode := meta["mode"]
	require.False(t, hasMode, "system meta carries no mode key")
}

func TestBuildSystemMetaOmitsEmpty(t *testing.T) {
	require.Empty(t, buildSystemMeta(&config.SystemConfig{}), "no metadata when all source fields are empty")
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

func TestLoadComponentsLoadsAgentBackgrounds(t *testing.T) {
	var gotCfg map[string]any
	backgrounds.Register("AgentFakeBG", func(cfg map[string]any) (backgrounds.Background, error) {
		gotCfg = cfg
		return &countingBackground{}, nil
	})
	llm.Register("AgentBgLLM", func(map[string]any) (llm.LLM, error) { return &fakeRuntimeLLM{}, nil })

	m := NewModeSetup(
		config.ModeConfig{
			Name:             "patrol",
			CortexLLM:        config.PluginSpec{Type: "AgentBgLLM"},
			AgentBackgrounds: []config.PluginSpec{{Type: "AgentFakeBG"}},
		},
		&config.SystemConfig{APIKey: "k"},
	)

	require.NoError(t, m.loadComponents())
	require.Len(t, m.agentBackgrounds, 1, "agent_backgrounds are loaded into the mode's background list")
	require.Equal(t, "patrol", gotCfg["mode"], "agent backgrounds are mode-scoped, so the mode key is injected")
}

func TestLoadGlobalBackgrounds(t *testing.T) {
	var gotCfg map[string]any
	backgrounds.Register("GlobalFakeBG", func(cfg map[string]any) (backgrounds.Background, error) {
		gotCfg = cfg
		return &countingBackground{}, nil
	})

	list, err := loadGlobalBackgrounds(&config.SystemConfig{
		APIKey:            "secret",
		GlobalBackgrounds: []config.PluginSpec{{Type: "GlobalFakeBG"}},
	})
	require.NoError(t, err)
	require.Len(t, list, 1)
	require.Equal(t, "secret", gotCfg["api_key"], "system api_key is injected as meta")
	_, hasMode := gotCfg["mode"]
	require.False(t, hasMode, "global backgrounds are mode-independent, so no mode key is injected")
}

func TestLoadGlobalBackgroundsEmpty(t *testing.T) {
	list, err := loadGlobalBackgrounds(&config.SystemConfig{})
	require.NoError(t, err)
	require.Empty(t, list, "no global backgrounds configured yields an empty list, not an error")
}

func TestLoadGlobalBackgroundsUnknown(t *testing.T) {
	_, err := loadGlobalBackgrounds(&config.SystemConfig{
		GlobalBackgrounds: []config.PluginSpec{{Type: "missing-global-bg"}},
	})
	require.Error(t, err)
	require.Contains(t, err.Error(), "global background", "the error is scoped to the global background")
}

type fakeRuntimeLLM struct{ schemas []map[string]any }

func (f *fakeRuntimeLLM) Call(context.Context, string, []llm.Message) (*llm.Response, error) {
	return &llm.Response{}, nil
}
func (f *fakeRuntimeLLM) SetSchemas(s []map[string]any)     { f.schemas = s }
func (f *fakeRuntimeLLM) FunctionSchemas() []map[string]any { return f.schemas }
