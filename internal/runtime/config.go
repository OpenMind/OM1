package runtime

import (
	"fmt"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/llm"
)

// modeSetup encapsulates the loaded components.
type modeSetup struct {
	cfg config.ModeConfig
	sys *config.SystemConfig

	sensors          []inputs.Sensor
	cortexLLM        llm.LLM
	agentActions     []*actions.AgentAction
	agentBackgrounds []backgrounds.Background
}

// NewModeSetup creates a new modeSetup for the given ModeConfig and SystemConfig.
func NewModeSetup(cfg config.ModeConfig, sys *config.SystemConfig) *modeSetup {
	return &modeSetup{cfg: cfg, sys: sys}
}

// loadComponents initializes all components for the mode based on its configuration.
func (m *modeSetup) loadComponents() error {
	modeName := m.cfg.Name
	meta := m.buildMeta(modeName)

	// Load inputs
	m.sensors = make([]inputs.Sensor, 0, len(m.cfg.AgentInputs))
	for _, spec := range m.cfg.AgentInputs {
		spec.Config = addMeta(spec.Config, meta)
		sensor, err := inputs.Load(spec.Type, spec.Config)
		if err != nil {
			return fmt.Errorf("input %q: %w", spec.Type, err)
		}
		m.sensors = append(m.sensors, sensor)
	}

	// Load LLM
	llmSpec := m.cfg.CortexLLM
	if llmSpec.Type == "" {
		llmSpec = m.sys.CortexLLM
	}
	if llmSpec.Type == "" {
		return fmt.Errorf("no LLM configured for mode %q (mode has no cortex_llm and no global cortex_llm is set)", modeName)
	}

	llmSpec.Config = addMeta(cloneConfig(llmSpec.Config), meta)
	var err error
	m.cortexLLM, err = llm.Load(llmSpec.Type, llmSpec.Config)
	if err != nil {
		return fmt.Errorf("load llm: %w", err)
	}

	// Load actions
	m.agentActions = make([]*actions.AgentAction, 0, len(m.cfg.AgentActions))
	for _, spec := range m.cfg.AgentActions {
		spec.Config = addMeta(spec.Config, meta)
		label := spec.LLMLabel
		if label == "" {
			label = spec.Name
		}
		agentAction, err := actions.Load(spec.Name, spec.Connector, label, spec.Config)
		if err != nil {
			return fmt.Errorf("action %q connector %q: %w", spec.Name, spec.Connector, err)
		}
		agentAction.ExcludeFromPrompt = spec.ExcludeFromPrompt
		m.agentActions = append(m.agentActions, agentAction)
	}

	// Load backgrounds
	m.agentBackgrounds = make([]backgrounds.Background, 0, len(m.cfg.AgentBackgrounds))
	for _, spec := range m.cfg.AgentBackgrounds {
		spec.Config = addMeta(spec.Config, meta)
		background, err := backgrounds.Load(spec.Type, spec.Config)
		if err != nil {
			return fmt.Errorf("background %q: %w", spec.Type, err)
		}
		m.agentBackgrounds = append(m.agentBackgrounds, background)
	}

	// Load lifecycle hooks
	for i := range m.cfg.LifecycleHooks {
		m.cfg.LifecycleHooks[i].HandlerConfig = addMeta(m.cfg.LifecycleHooks[i].HandlerConfig, meta)
	}

	return nil
}

// toRuntimeConfig converts the modeSetup to a RuntimeConfig for execution.
func (m *modeSetup) toRuntimeConfig() *config.RuntimeConfig {
	rc := &config.RuntimeConfig{
		Version:          m.sys.Version,
		Name:             m.sys.Name,
		Hertz:            m.cfg.Hertz,
		SystemPromptBase: mergePrompt(m.sys.SystemPromptBase, m.cfg.SystemPromptBase),
		SystemGovernance: m.sys.SystemGovernance,
		PromptExamples:   m.sys.PromptExamples,
		KnowledgeBase:    m.sys.KnowledgeBase,
		ActionExecMode:   m.cfg.ActionExecMode,
		ActionDeps:       m.cfg.ActionDeps,
		UseTracer:        m.sys.UseTracer,
	}
	if rc.Hertz == 0 {
		rc.Hertz = 1.0
	}
	return rc
}

// buildMeta constructs a metadata map for the mode by combining system-level metadata with the mode name.
func (m *modeSetup) buildMeta(modeName string) map[string]any {
	meta := buildSystemMeta(m.sys)
	if modeName != "" {
		meta["mode"] = modeName
	}
	return meta
}

// buildSystemMeta constructs a metadata map from the system configuration, including API key, robot IP, URID, and simulation flag.
func buildSystemMeta(sys *config.SystemConfig) map[string]any {
	meta := make(map[string]any)
	if sys.APIKey != "" {
		meta["api_key"] = sys.APIKey
	}
	if sys.RobotIP != "" {
		meta["robot_ip"] = sys.RobotIP
	}
	if sys.URID != "" {
		meta["URID"] = sys.URID
	}
	if sys.UseSim {
		meta["use_sim"] = true
	}
	return meta
}

// loadGlobalBackgrounds initializes the global background tasks based on the system configuration.
func loadGlobalBackgrounds(sys *config.SystemConfig) ([]backgrounds.Background, error) {
	meta := buildSystemMeta(sys)
	list := make([]backgrounds.Background, 0, len(sys.GlobalBackgrounds))
	for _, spec := range sys.GlobalBackgrounds {
		spec.Config = addMeta(cloneConfig(spec.Config), meta)
		background, err := backgrounds.Load(spec.Type, spec.Config)
		if err != nil {
			return nil, fmt.Errorf("global background %q: %w", spec.Type, err)
		}
		list = append(list, background)
	}
	return list, nil
}

// cloneConfig returns a shallow copy of a config map so that callers can add
// per-mode metadata without mutating a shared (e.g. global) source map.
func cloneConfig(cfg map[string]any) map[string]any {
	clone := make(map[string]any, len(cfg))
	for k, v := range cfg {
		clone[k] = v
	}
	return clone
}

// addMeta adds metadata key-value pairs to a config map if they are not already present.
func addMeta(cfg map[string]any, meta map[string]any) map[string]any {
	if cfg == nil {
		cfg = make(map[string]any)
	}
	for k, v := range meta {
		if _, exists := cfg[k]; !exists {
			cfg[k] = v
		}
	}
	return cfg
}

// collectSchemas extracts the schemas from agent actions that are not excluded from the prompt.
func collectSchemas(agentActions []*actions.AgentAction) []map[string]any {
	schemas := make([]map[string]any, 0, len(agentActions))

	for _, agentAction := range agentActions {
		if !agentAction.ExcludeFromPrompt && agentAction.Schema != nil {
			schemas = append(schemas, agentAction.Schema)
		}
	}

	return schemas
}

// mergePrompt combines the global system prompt base with the mode-specific one,
// giving precedence to the mode-specific prompt if it exists.
func mergePrompt(global, modeSpecific string) string {
	if modeSpecific != "" {
		return modeSpecific
	}
	return global
}

// toolCallsToMaps converts LLM tool calls to map representations for action orchestration.
func toolCallsToMaps(toolCalls []llm.ToolCall) []map[string]any {
	result := make([]map[string]any, len(toolCalls))
	for i, tc := range toolCalls {
		result[i] = map[string]any{
			"name":      tc.Name,
			"arguments": tc.Arguments,
		}
	}
	return result
}

// traceOutput converts an LLM response's tool calls into a slice of maps for structured tracing.
func traceOutput(response *llm.Response) []map[string]any {
	out := make([]map[string]any, 0, len(response.ToolCalls))
	for _, tc := range response.ToolCalls {
		out = append(out, map[string]any{
			"type":  tc.Name,
			"value": tc.Arguments,
		})
	}
	return out
}
