package config

import (
	"encoding/json"
	"fmt"
)

// SystemConfig is the top-level, file-level configuration.
type SystemConfig struct {
	Version string `json:"version"`
	Name    string `json:"name"`

	Hertz             float64      `json:"hertz"`
	SystemPromptBase  string       `json:"system_prompt_base"`
	SystemGovernance  string       `json:"system_governance"`
	PromptExamples    string       `json:"system_prompt_examples"`
	AgentInputs       []PluginSpec `json:"agent_inputs"`
	CortexLLM         PluginSpec   `json:"cortex_llm"`
	AgentActions      []ActionSpec `json:"agent_actions"`
	AgentBackgrounds  []PluginSpec `json:"agent_backgrounds"`
	GlobalBackgrounds []PluginSpec `json:"global_backgrounds"`
	MCPServers        []MCPSpec    `json:"mcp_servers"`
	KnowledgeBase     *KBSpec      `json:"knowledge_base"`
	Memory            *MemorySpec  `json:"memory"`
	LifecycleHooks    []HookSpec   `json:"lifecycle_hooks"`

	APIKey  string `json:"api_key"`
	RobotIP string `json:"robot_ip"`
	URID    string `json:"URID"`

	UseSim bool `json:"use_sim"`

	UseTracer *TracerConfig `json:"use_tracer"`

	DefaultMode       string                `json:"default_mode"`
	AllowManualSwitch bool                  `json:"allow_manual_switching"`
	ModeMemoryEnabled bool                  `json:"mode_memory_enabled"`
	Modes             map[string]ModeConfig `json:"modes"`
	TransitionRules   []TransitionRule      `json:"transition_rules"`
	GlobalHooks       []HookSpec            `json:"global_lifecycle_hooks"`
}

// ModeConfig describes one operating mode in a multi-mode system.
type ModeConfig struct {
	Name             string       `json:"name"`
	DisplayName      string       `json:"display_name"`
	Description      string       `json:"description"`
	Hertz            float64      `json:"hertz"`
	SystemPromptBase string       `json:"system_prompt_base"`
	AgentInputs      []PluginSpec `json:"agent_inputs"`
	CortexLLM        PluginSpec   `json:"cortex_llm"`
	AgentActions     []ActionSpec `json:"agent_actions"`
	AgentBackgrounds []PluginSpec `json:"agent_backgrounds"`
	MCPServers       []MCPSpec    `json:"mcp_servers"`
	LifecycleHooks   []HookSpec   `json:"lifecycle_hooks"`
	TimeoutSeconds   float64      `json:"timeout_seconds"`

	// Execution mode: "concurrent" | "sequential" | "dependencies"
	ActionExecMode string              `json:"action_execution_mode"`
	ActionDeps     map[string][]string `json:"action_dependencies"`
}

// PluginSpec references a plugin by type + opaque config map.
type PluginSpec struct {
	Type   string         `json:"type"`
	Config map[string]any `json:"config"`
}

// ActionSpec extends PluginSpec with action-specific metadata.
type ActionSpec struct {
	Name              string         `json:"name"`
	LLMLabel          string         `json:"llm_label"`
	Implementation    string         `json:"implementation"`
	Connector         string         `json:"connector"`
	Config            map[string]any `json:"config"`
	ExcludeFromPrompt bool           `json:"exclude_from_prompt"`
}

// TransitionRule defines when/how modes switch.
type TransitionRule struct {
	FromMode          string         `json:"from_mode"`
	ToMode            string         `json:"to_mode"`
	TransitionType    string         `json:"transition_type"` // "input_triggered" | "time_based" | "context_aware"
	TriggerKeywords   []string       `json:"trigger_keywords"`
	TimeoutSeconds    float64        `json:"timeout_seconds"`
	CooldownSeconds   float64        `json:"cooldown_seconds"`
	Priority          int            `json:"priority"`
	ContextConditions map[string]any `json:"context_conditions"`
}

// HookSpec declares a lifecycle hook.
type HookSpec struct {
	HookType      string         `json:"hook_type"`    // "on_entry" | "on_exit" | "on_startup" | "on_timeout"
	HandlerType   string         `json:"handler_type"` // "action" | "message" | "command"
	HandlerConfig map[string]any `json:"handler_config"`
}

// MCPSpec describes one MCP server.
type MCPSpec struct {
	Name      string            `json:"name"`
	Transport string            `json:"transport"`
	Command   string            `json:"command"`
	Args      []string          `json:"args"`
	Env       map[string]string `json:"env"`
	URL       string            `json:"url"`
	Headers   map[string]string `json:"headers"`
}

// KBSpec describes the optional knowledge base (RAG).
type KBSpec struct {
	Name     string  `json:"knowledge_base_name"`
	Root     string  `json:"knowledge_base_root"`
	BaseURL  string  `json:"base_url"`
	TopK     int     `json:"top_k"`
	MinScore float64 `json:"min_score"`
}

// TracerConfig enables OM1's LLM interaction tracer (writes traces/<date>.jsonl, feeds any in-process subscriber).
// Accepts either a plain bool (use_tracer: true, the pre-existing form) or an object with quality_scorer nested in it.
type TracerConfig struct {
	Enabled bool `json:"enabled"`

	// QualityScorer is nested here since it depends entirely on tracing being on.
	QualityScorer *QualityScorerConfig `json:"quality_scorer"`
}

// UnmarshalJSON lets use_tracer be written as either a bool or an object -- see TracerConfig's doc comment.
func (t *TracerConfig) UnmarshalJSON(data []byte) error {
	var enabled bool
	if err := json.Unmarshal(data, &enabled); err == nil {
		t.Enabled = enabled
		return nil
	}
	type alias TracerConfig // avoid recursing back into this method
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return fmt.Errorf("use_tracer must be a bool or an object: %w", err)
	}
	*t = TracerConfig(a)
	return nil
}

// QualityScorerConfig describes the optional live conversation-quality scorer, scraped as Prometheus metrics.
type QualityScorerConfig struct {
	Enabled bool   `json:"enabled"`
	Model   string `json:"model"`
	BaseURL string `json:"base_url"`

	// APIKey overrides the top-level system api_key; falls back to it if unset (see cmd/main.go).
	APIKey string `json:"api_key"`
}

// MemorySpec describes the optional long-term memory system.
type MemorySpec struct {
	Enabled         bool `json:"enabled"`
	CloudConnection bool `json:"cloud_connection"`
}

// RuntimeConfig is the resolved, instantiated state for the active mode.
// Components are fully constructed and ready to run.
type RuntimeConfig struct {
	Version          string
	Name             string
	Hertz            float64
	SystemPromptBase string
	SystemGovernance string
	PromptExamples   string
	KnowledgeBase    *KBSpec
	ActionExecMode   string
	ActionDeps       map[string][]string
}
