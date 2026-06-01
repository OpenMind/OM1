package runtime

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/hooks"
	"github.com/openmind/om1/internal/util"
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

	// userContext holds contextual information used to evaluate context-aware transitions.
	userContext map[string]any
}

// NewModeManager creates a new ModeManager with the given system configuration and logger.
func NewModeManager(systemConfig *config.SystemConfig, log *zap.Logger) *ModeManager {
	manager := &ModeManager{
		systemConfig: systemConfig,
		cooldowns:    make(map[string]time.Time),
		log:          log,
		statePath:    filepath.Join("config", "memory", ".mode_state.json"),
		globalHooks:  hooks.NewHooks(systemConfig.GlobalHooks, log),
		userContext:  make(map[string]any),
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

// Close releases resources held by the manager.
func (m *ModeManager) Close() {}

// UpdateUserContext merges the given key/value pairs into the user context used
// for context-aware transitions.
func (m *ModeManager) UpdateUserContext(update map[string]any) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for k, v := range update {
		m.userContext[k] = v
	}
}

func (m *ModeManager) CurrentMode() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state.CurrentMode
}

// CheckTransitions evaluates all transition rules for the current mode and
// returns the target mode for the first category that fires, or "" if none do.
func (m *ModeManager) CheckTransitions(ctx context.Context, latestInputs []string) string {
	if target := m.checkTimeBased(ctx); target != "" {
		m.log.Info("time-based transition", zap.String("to", target))
		return target
	}
	if target := m.checkContextAware(); target != "" {
		m.log.Info("context-aware transition", zap.String("to", target))
		return target
	}
	if target := m.checkInputTriggered(latestInputs); target != "" {
		m.log.Info("input-triggered transition", zap.String("to", target))
		return target
	}

	return ""
}

// checkTimeBased returns the target mode if the current mode has been active
// long enough to satisfy a time-based rule, firing OnTimeout hooks first.
func (m *ModeManager) checkTimeBased(ctx context.Context) string {
	m.mu.RLock()
	currentMode := m.state.CurrentMode
	modeStartTime := m.state.ModeStartTime
	m.mu.RUnlock()

	elapsed := time.Since(modeStartTime).Seconds()

	for _, rule := range m.systemConfig.TransitionRules {
		if rule.FromMode != currentMode && rule.FromMode != "*" {
			continue
		}
		if rule.TransitionType != "time_based" {
			continue
		}
		if rule.TimeoutSeconds <= 0 || elapsed < rule.TimeoutSeconds {
			continue
		}
		if !m.cooldownExpired(rule) {
			continue
		}
		if _, ok := m.systemConfig.Modes[rule.ToMode]; !ok {
			continue
		}

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
	return ""
}

// checkContextAware returns the target mode of the highest-priority context-aware rule
// whose conditions match the user context and cooldowns have expired, or "" if none match.
func (m *ModeManager) checkContextAware() string {
	m.mu.RLock()
	currentMode := m.state.CurrentMode
	userCtx := make(map[string]any, len(m.userContext))
	for k, v := range m.userContext {
		userCtx[k] = v
	}
	m.mu.RUnlock()

	var matching []config.TransitionRule
	for _, rule := range m.systemConfig.TransitionRules {
		if rule.FromMode != currentMode && rule.FromMode != "*" {
			continue
		}
		if rule.TransitionType != "context_aware" {
			continue
		}
		if !m.cooldownExpired(rule) {
			continue
		}
		if _, ok := m.systemConfig.Modes[rule.ToMode]; !ok {
			continue
		}
		if evaluateContextConditions(rule.ContextConditions, userCtx) {
			matching = append(matching, rule)
		}
	}
	return highestPriorityTarget(matching)
}

// checkInputTriggered returns the highest-priority input-triggered rule whose
// trigger keywords appear in the latest inputs.
func (m *ModeManager) checkInputTriggered(latestInputs []string) string {
	combined := strings.ToLower(joinStrings(latestInputs))
	if strings.TrimSpace(combined) == "" {
		return ""
	}

	m.mu.RLock()
	currentMode := m.state.CurrentMode
	m.mu.RUnlock()

	var matching []config.TransitionRule
	for _, rule := range m.systemConfig.TransitionRules {
		if rule.FromMode != currentMode && rule.FromMode != "*" {
			continue
		}
		if rule.TransitionType != "input_triggered" {
			continue
		}
		if !m.cooldownExpired(rule) {
			continue
		}
		if _, ok := m.systemConfig.Modes[rule.ToMode]; !ok {
			continue
		}
		for _, keyword := range rule.TriggerKeywords {
			if keyword != "" && strings.Contains(combined, strings.ToLower(keyword)) {
				matching = append(matching, rule)
				break
			}
		}
	}
	return highestPriorityTarget(matching)
}

// highestPriorityTarget returns the ToMode of the highest-priority rule, or ""
// if the slice is empty. Stable sort keeps configuration order among equal
// priorities, matching Python's sort(..., reverse=True) on a preserved list.
func highestPriorityTarget(rules []config.TransitionRule) string {
	if len(rules) == 0 {
		return ""
	}
	sort.SliceStable(rules, func(i, j int) bool {
		return rules[i].Priority > rules[j].Priority
	})
	return rules[0].ToMode
}

// evaluateContextConditions reports whether every condition in conds is
// satisfied by userCtx. An empty condition set always matches.
func evaluateContextConditions(conds, userCtx map[string]any) bool {
	for key, expected := range conds {
		if !evaluateSingleCondition(key, expected, userCtx) {
			return false
		}
	}
	return true
}

// evaluateSingleCondition checks whether a single condition is satisfied by the user context.
func evaluateSingleCondition(key string, expected any, userCtx map[string]any) bool {
	actual, ok := userCtx[key]
	if !ok {
		return false
	}

	switch exp := expected.(type) {
	case map[string]any:
		minVal, hasMin := exp["min"]
		maxVal, hasMax := exp["max"]
		if hasMin || hasMax {
			actualNum, ok := util.ToFloat(actual)
			if !ok {
				return false
			}
			if hasMin {
				if minNum, ok := util.ToFloat(minVal); ok && actualNum < minNum {
					return false
				}
			}
			if hasMax {
				if maxNum, ok := util.ToFloat(maxVal); ok && actualNum > maxNum {
					return false
				}
			}
			return true
		}
		if c, ok := exp["contains"]; ok {
			actualStr, ok1 := actual.(string)
			condStr, ok2 := c.(string)
			if !ok1 || !ok2 {
				return false
			}
			return strings.Contains(strings.ToLower(actualStr), strings.ToLower(condStr))
		}
		if oneOf, ok := exp["one_of"]; ok {
			if list, ok := oneOf.([]any); ok {
				for _, item := range list {
					if reflect.DeepEqual(actual, item) {
						return true
					}
				}
			}
			return false
		}
		if notVal, ok := exp["not"]; ok {
			return !reflect.DeepEqual(actual, notVal)
		}
		return false

	case []any:
		for _, item := range exp {
			if reflect.DeepEqual(actual, item) {
				return true
			}
		}
		return false

	default:
		return reflect.DeepEqual(actual, expected)
	}
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
