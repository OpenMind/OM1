package runtime

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/hooks"
)

// ModeState represents the current state of the system's mode.
type ModeState struct {
	CurrentMode       string    `json:"current_mode"`
	PreviousMode      string    `json:"previous_mode"`
	ModeStartTime     time.Time `json:"mode_start_time"`
	TransitionHistory []string  `json:"transition_history"`
}

// ModeManager manages the current mode of the system.
type ModeManager struct {
	mu           sync.RWMutex
	state        ModeState
	systemConfig *config.SystemConfig
	cooldowns    map[string]time.Time
	log          *zap.Logger
	statePath    string
	globalHooks  *hooks.Runner
}

// NewModeManager creates a new ModeManager with the given system configuration and logger.
func NewModeManager(systemConfig *config.SystemConfig, log *zap.Logger) *ModeManager {
	manager := &ModeManager{
		systemConfig: systemConfig,
		cooldowns:    make(map[string]time.Time),
		log:          log,
		statePath:    filepath.Join("config", "memory", ".mode_state.json"),
		globalHooks:  hooks.NewHooks(systemConfig.GlobalHooks, log),
	}
	manager.state = ModeState{
		CurrentMode:   systemConfig.DefaultMode,
		ModeStartTime: time.Now(),
	}
	if systemConfig.ModeMemoryEnabled {
		manager.load()
	}
	return manager
}

func (m *ModeManager) CurrentMode() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state.CurrentMode
}

func (m *ModeManager) CheckTransitions(ctx context.Context, latestInputs []string) string {
	m.mu.RLock()
	currentMode := m.state.CurrentMode
	modeStartTime := m.state.ModeStartTime
	m.mu.RUnlock()

	for _, rule := range m.systemConfig.TransitionRules {
		if rule.FromMode != currentMode && rule.FromMode != "*" {
			continue
		}
		if !m.cooldownExpired(rule) {
			continue
		}

		switch rule.TransitionType {
		case "time_based":
			elapsed := time.Since(modeStartTime).Seconds()
			if rule.TimeoutSeconds > 0 && elapsed >= rule.TimeoutSeconds {
				if modeCfg, ok := m.systemConfig.Modes[currentMode]; ok {
					timeoutCtx := map[string]any{
						"mode_name":       currentMode,
						"timeout_seconds": rule.TimeoutSeconds,
						"actual_duration": elapsed,
						"timestamp":       float64(time.Now().UnixMilli()) / 1000.0,
					}
					if err := hooks.NewHooks(modeCfg.LifecycleHooks, m.log).Run(ctx, hooks.OnTimeout, timeoutCtx); err != nil {
						m.log.Warn("mode OnTimeout hook failed during mode transition",
							zap.String("mode", currentMode),
							zap.Float64("timeout_seconds", rule.TimeoutSeconds),
							zap.Float64("actual_duration", elapsed),
							zap.Error(err),
						)
					}
				}
				return rule.ToMode
			}

		case "input_triggered":
			combinedInput := joinStrings(latestInputs)
			for _, keyword := range rule.TriggerKeywords {
				if containsKeyword(combinedInput, keyword) {
					return rule.ToMode
				}
			}
		}
	}
	return ""
}

// Transition updates the mode state and fires lifecycle hooks for the transition.
func (m *ModeManager) Transition(toMode, reason string, exitHooks, entryHooks *hooks.Runner) {
	fromMode := m.state.CurrentMode
	transitionKey := fromMode + "->" + toMode
	transitionCtx := map[string]any{
		"from_mode":      fromMode,
		"to_mode":        toMode,
		"reason":         reason,
		"timestamp":      float64(time.Now().UnixMilli()) / 1000.0,
		"transition_key": transitionKey,
	}

	if exitHooks != nil {
		if err := exitHooks.Run(context.Background(), hooks.OnExit, transitionCtx); err != nil {
			m.log.Warn("mode OnExit hook failed during mode transition",
				zap.String("from", fromMode),
				zap.String("to", toMode),
				zap.Error(err),
			)
		}
	}

	if err := m.globalHooks.Run(context.Background(), hooks.OnExit, transitionCtx); err != nil {
		m.log.Warn("global OnExit hook failed during mode transition",
			zap.String("from", fromMode),
			zap.String("to", toMode),
			zap.Error(err),
		)
	}

	m.mu.Lock()
	cooldownKey := m.state.CurrentMode + "→" + toMode
	m.cooldowns[cooldownKey] = time.Now()

	m.state.TransitionHistory = append(m.state.TransitionHistory, m.state.CurrentMode)
	if len(m.state.TransitionHistory) > 20 {
		m.state.TransitionHistory = m.state.TransitionHistory[1:]
	}
	m.state.PreviousMode = m.state.CurrentMode
	m.state.CurrentMode = toMode
	m.state.ModeStartTime = time.Now()

	m.log.Info("mode transition",
		zap.String("from", m.state.PreviousMode),
		zap.String("to", toMode),
	)

	if m.systemConfig.ModeMemoryEnabled {
		m.save()
	}
	m.mu.Unlock()

	if err := m.globalHooks.Run(context.Background(), hooks.OnEntry, transitionCtx); err != nil {
		m.log.Warn("global OnEntry hook failed during mode transition",
			zap.String("from", fromMode),
			zap.String("to", toMode),
			zap.Error(err),
		)
	}

	if entryHooks != nil {
		if err := entryHooks.Run(context.Background(), hooks.OnEntry, transitionCtx); err != nil {
			m.log.Warn("mode OnEntry hook failed during mode transition",
				zap.String("from", fromMode),
				zap.String("to", toMode),
				zap.Error(err),
			)
		}
	}
}

func (m *ModeManager) cooldownExpired(rule config.TransitionRule) bool {
	if rule.CooldownSeconds <= 0 {
		return true
	}
	cooldownKey := rule.FromMode + "→" + rule.ToMode
	lastTransition, ok := m.cooldowns[cooldownKey]
	if !ok {
		return true
	}
	return time.Since(lastTransition).Seconds() >= rule.CooldownSeconds
}

func (m *ModeManager) load() {
	data, err := os.ReadFile(m.statePath)
	if err != nil {
		return
	}
	var savedState ModeState
	if err := json.Unmarshal(data, &savedState); err == nil {
		if _, ok := m.systemConfig.Modes[savedState.CurrentMode]; ok {
			m.state = savedState
		}
	}
}

func (m *ModeManager) save() {
	data, err := json.Marshal(m.state)
	if err != nil {
		return
	}
	_ = os.MkdirAll(filepath.Dir(m.statePath), 0o755)
	_ = os.WriteFile(m.statePath, data, 0o644)
}

func joinStrings(parts []string) string {
	result := ""
	for _, part := range parts {
		result += " " + part
	}
	return result
}

func containsKeyword(text, keyword string) bool {
	return len(text) > 0 && len(keyword) > 0 &&
		(len(text) >= len(keyword)) &&
		(func() bool {
			for i := 0; i <= len(text)-len(keyword); i++ {
				if text[i:i+len(keyword)] == keyword {
					return true
				}
			}
			return false
		}())
}
