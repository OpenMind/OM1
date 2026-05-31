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

	sensors        []inputs.Sensor
	cortexLLM      llm.LLM
	agentActions   []*actions.AgentAction
	backgroundList []backgrounds.Background
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
	m.backgroundList = make([]backgrounds.Background, 0, len(m.cfg.Backgrounds))
	for _, spec := range m.cfg.Backgrounds {
		spec.Config = addMeta(spec.Config, meta)
		background, err := backgrounds.Load(spec.Type, spec.Config)
		if err != nil {
			return fmt.Errorf("background %q: %w", spec.Type, err)
		}
		m.backgroundList = append(m.backgroundList, background)
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
	}
	if rc.Hertz == 0 {
		rc.Hertz = 1.0
	}
	return rc
}

// buildMeta constructs a metadata map for the mode, including global config values and the mode name.
func (m *modeSetup) buildMeta(modeName string) map[string]string {
	meta := make(map[string]string)
	if m.sys.APIKey != "" {
		meta["api_key"] = m.sys.APIKey
	}
	if m.sys.RobotIP != "" {
		meta["robot_ip"] = m.sys.RobotIP
	}
	if m.sys.URID != "" {
		meta["URID"] = m.sys.URID
	}
	if modeName != "" {
		meta["mode"] = modeName
	}
	return meta
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
func addMeta(cfg map[string]any, meta map[string]string) map[string]any {
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
