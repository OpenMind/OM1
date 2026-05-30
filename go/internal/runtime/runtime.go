package runtime

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/actions"
	"github.com/openmind/om1/internal/backgrounds"
	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/fuser"
	"github.com/openmind/om1/internal/hooks"
	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/providers"
)

type Options struct {
	HotReload     bool
	CheckInterval float64
}

type modeState struct {
	runtimeConfig      *config.RuntimeConfig
	promptFuser        *fuser.Fuser
	cortexLLM          *llm.Orchestrator
	actionOrchestrator *actions.Orchestrator
	bgOrchestrator     *backgrounds.Orchestrator // nil when mode has no backgrounds
	sensors            []inputs.Sensor           // stored here; InputOrchestrator created in startOrchestrators
	inputOrchestrator  *inputs.Orchestrator      // set by startOrchestrators
	modeHooks          *hooks.Runner

	cancelCtx      context.CancelFunc
	inputDone      <-chan struct{}
	actionDone     <-chan struct{}
	backgroundDone <-chan struct{}
	cortexDone     <-chan struct{}
}

type Runtime struct {
	systemConfig *config.SystemConfig
	opts         Options
	log          *zap.Logger
	manager      *ModeManager
	ioProvider   *providers.IOProvider

	mu                        sync.Mutex
	current                   *modeState
	isReloading               bool
	modeTransitionHandlerOnce bool // ensures handleModeTransitions goroutine starts only once

	modeTransitionCh chan string
}

func New(systemConfig *config.SystemConfig, log *zap.Logger, opts Options) *Runtime {
	return &Runtime{
		systemConfig:     systemConfig,
		opts:             opts,
		log:              log,
		manager:          NewModeManager(systemConfig, log),
		ioProvider:       providers.IO(),
		modeTransitionCh: make(chan string, 1),
	}
}

func (rt *Runtime) Run(ctx context.Context) error {
	if rt.opts.HotReload {
		go rt.watchConfig(ctx)
	}

	initialMode := rt.manager.CurrentMode()
	if err := rt.initializeMode(initialMode); err != nil {
		return fmt.Errorf("initialize mode %q: %w", initialMode, err)
	}

	rt.mu.Lock()
	current := rt.current
	rt.mu.Unlock()

	if current != nil {
		startupCtx := map[string]any{
			"mode_name":   initialMode,
			"system_name": rt.systemConfig.Name,
			"timestamp":   float64(time.Now().UnixMilli()) / 1000.0,
		}

		if err := rt.manager.globalHooks.Run(ctx, hooks.OnStartup, startupCtx); err != nil {
			rt.log.Warn("global startup hook failed", zap.Error(err))
		}

		if err := current.modeHooks.Run(ctx, hooks.OnStartup, startupCtx); err != nil {
			rt.log.Warn("mode startup hook failed", zap.Error(err))
		}
	}

	rt.startOrchestrators(ctx)

	if current != nil {
		entryCtx := map[string]any{
			"mode_name":   initialMode,
			"system_name": rt.systemConfig.Name,
			"timestamp":   float64(time.Now().UnixMilli()) / 1000.0,
		}

		if err := rt.manager.globalHooks.Run(ctx, hooks.OnEntry, entryCtx); err != nil {
			rt.log.Warn("global entry hook failed", zap.Error(err))
		}

		if err := current.modeHooks.Run(ctx, hooks.OnEntry, entryCtx); err != nil {
			rt.log.Warn("mode entry hook failed", zap.Error(err))
		}
	}

	<-ctx.Done()

	rt.stopOrchestrators()
	return ctx.Err()
}

func (rt *Runtime) initializeMode(modeName string) error {
	modeCfg, ok := rt.systemConfig.Modes[modeName]
	if !ok {
		return fmt.Errorf("mode %q not found in config", modeName)
	}

	modeConfig := NewModeSetup(modeCfg, rt.systemConfig)

	if err := modeConfig.loadComponents(); err != nil {
		return err
	}

	runtimeConfig := modeConfig.toRuntimeConfig()

	rt.log.Info("initializing mode", zap.String("mode", modeCfg.DisplayName))

	state := &modeState{
		runtimeConfig: runtimeConfig,
		promptFuser:   fuser.NewFuser(runtimeConfig, modeConfig.agentActions, nil),
		cortexLLM: llm.NewOrchestrator(
			modeConfig.cortexLLM,
			modeCfg.CortexLLM.Config,
			collectSchemas(modeConfig.agentActions),
		),
		actionOrchestrator: actions.NewOrchestrator(
			modeConfig.agentActions,
			actions.ExecMode(runtimeConfig.ActionExecMode),
			runtimeConfig.ActionDeps,
			rt.log,
		),
		sensors:   modeConfig.sensors,
		modeHooks: hooks.NewHooks(modeCfg.LifecycleHooks, rt.log),
	}
	if len(modeConfig.backgroundList) > 0 {
		state.bgOrchestrator = backgrounds.NewOrchestrator(modeConfig.backgroundList, rt.log)
	}

	rt.mu.Lock()
	rt.current = state
	rt.mu.Unlock()

	rt.log.Info("mode initialised", zap.String("mode", modeName))
	return nil
}

// startOrchestrators mirrors Python's _start_orchestrators: it creates the
// InputOrchestrator, starts all goroutine pools, starts the cortex loop, and
// (on first call) starts the long-lived handleModeTransitions goroutine.
func (rt *Runtime) startOrchestrators(ctx context.Context) {
	if ctx.Err() != nil {
		return
	}

	rt.mu.Lock()
	current := rt.current
	rt.mu.Unlock()

	if current == nil {
		return
	}

	modeCtx, cancel := context.WithCancel(ctx)
	current.cancelCtx = cancel

	// Create InputOrchestrator here, mirroring Python which does
	// self.input_orchestrator = InputOrchestrator(self.current_config.agent_inputs)
	// inside _start_orchestrators, not _initialize_mode.
	current.inputOrchestrator = inputs.NewOrchestrator(current.sensors, rt.log)
	current.inputDone = current.inputOrchestrator.Start(modeCtx)

	// Mirror Python: if self.action_orchestrator / if self.background_orchestrator
	if current.actionOrchestrator != nil {
		current.actionDone = current.actionOrchestrator.Start(modeCtx)
	}
	if current.bgOrchestrator != nil {
		current.backgroundDone = current.bgOrchestrator.Start(modeCtx)
	}

	cortexDone := make(chan struct{})
	current.cortexDone = cortexDone
	go func() {
		defer close(cortexDone)
		rt.runCortexLoop(modeCtx)
	}()

	// Start the mode transition handler goroutine once for the process lifetime,
	// mirroring Python's: if not self.mode_transition_task or self.mode_transition_task.done()
	rt.mu.Lock()
	if !rt.modeTransitionHandlerOnce {
		rt.modeTransitionHandlerOnce = true
		go rt.handleModeTransitions(ctx)
	}
	rt.mu.Unlock()
}

// stopOrchestrators cancels the current mode context and waits for all goroutine
// pools to finish (up to 15 seconds total), then runs OnExit hooks.
// It mirrors Python's _stop_current_orchestrators.
func (rt *Runtime) stopOrchestrators() {
	rt.mu.Lock()
	current := rt.current
	rt.current = nil
	rt.mu.Unlock()

	if current == nil || current.cancelCtx == nil {
		return
	}

	current.cancelCtx()

	stopCtx, stopCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer stopCancel()

	type namedCh struct {
		name string
		ch   <-chan struct{}
	}
	pools := []namedCh{
		{"cortex", current.cortexDone},
		{"action", current.actionDone},
		{"background", current.backgroundDone},
		{"input", current.inputDone},
	}

	// Wait for async goroutine pools concurrently under a shared 15-second timeout,
	// mirroring Python's asyncio.wait(..., timeout=15.0, return_when=ALL_COMPLETED).
	var wg sync.WaitGroup
	for _, p := range pools {
		if p.ch == nil {
			continue
		}
		wg.Add(1)
		go func(name string, ch <-chan struct{}) {
			defer wg.Done()
			select {
			case <-ch:
			case <-stopCtx.Done():
				rt.log.Warn("orchestrator shutdown timed out", zap.String("pool", name))
			}
		}(p.name, p.ch)
	}

	wg.Wait()
}

// onModeTransition stops the current mode's orchestrators, initialises the new
// mode and restarts all orchestrators, mirroring Python's _on_mode_transition.
func (rt *Runtime) onModeTransition(ctx context.Context, fromMode, toMode string) error {
	rt.log.Info("handling mode transition",
		zap.String("from", fromMode),
		zap.String("to", toMode),
	)

	rt.mu.Lock()
	rt.isReloading = true
	rt.mu.Unlock()
	defer func() {
		rt.mu.Lock()
		rt.isReloading = false
		rt.mu.Unlock()
	}()

	// Capture the departing mode's hooks before stopOrchestrators clears current.
	rt.mu.Lock()
	var exitHooks *hooks.Runner
	if rt.current != nil {
		exitHooks = rt.current.modeHooks
	}
	rt.mu.Unlock()

	rt.stopOrchestrators()

	if err := rt.initializeMode(toMode); err != nil {
		return fmt.Errorf("initialize mode %q: %w", toMode, err)
	}

	// Capture the arriving mode's hooks after initializeMode sets the new current.
	rt.mu.Lock()
	var entryHooks *hooks.Runner
	if rt.current != nil {
		entryHooks = rt.current.modeHooks
	}
	rt.mu.Unlock()

	rt.manager.Transition(toMode, "transition", exitHooks, entryHooks)
	rt.startOrchestrators(ctx)

	rt.log.Info("mode transition complete", zap.String("to", toMode))
	return nil
}

// handleModeTransitions processes queued mode transition requests in a
// dedicated goroutine, mirroring Python's _handle_mode_transitions.
func (rt *Runtime) handleModeTransitions(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case toMode := <-rt.modeTransitionCh:
			fromMode := rt.manager.CurrentMode()
			if err := rt.onModeTransition(ctx, fromMode, toMode); err != nil {
				rt.log.Error("mode transition failed",
					zap.String("to", toMode),
					zap.Error(err),
				)
			}
		}
	}
}

// runCortexLoop is the main processing loop for the current mode, mirroring
// Python's _run_cortex_loop. It is started as a goroutine by startOrchestrators
// and exits when modeCtx is cancelled.
func (rt *Runtime) runCortexLoop(ctx context.Context) {
	modeName := rt.manager.CurrentMode()
	rt.log.Info("cortex loop started", zap.String("mode", modeName))

	rt.mu.Lock()
	current := rt.current
	rt.mu.Unlock()

	if current == nil {
		return
	}

	tickInterval := time.Duration(float64(time.Second) / current.runtimeConfig.Hertz)
	ticker := time.NewTicker(tickInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			rt.log.Info("cortex loop exiting", zap.String("mode", modeName))
			return
		case tickStart := <-ticker.C:
			rt.tick(ctx, current, tickStart)
		case <-current.inputOrchestrator.TickNow():
			select {
			case <-ticker.C:
			default:
			}
			ticker.Reset(tickInterval)
			rt.tick(ctx, current, time.Now())
		}
	}
}

// tick executes a single cortex cycle: checks for mode transitions, fuses a
// prompt, calls the LLM, executes tool calls and records telemetry.
// It mirrors Python's _tick.
func (rt *Runtime) tick(ctx context.Context, current *modeState, tickStart time.Time) {
	if ctx.Err() != nil {
		return
	}

	rt.mu.Lock()
	reloading := rt.isReloading
	rt.mu.Unlock()

	if reloading {
		rt.log.Debug("skipping tick during mode transition")
		return
	}

	sensorBuffers := current.inputOrchestrator.Buffers()

	nextMode := rt.manager.CheckTransitions(ctx, sensorBuffers)
	if nextMode != "" {
		select {
		case rt.modeTransitionCh <- nextMode:
			rt.log.Info("mode transition scheduled", zap.String("to", nextMode))
		default:
		}
		return
	}

	prompt, err := current.promptFuser.Fuse(ctx, sensorBuffers)
	if err != nil {
		rt.log.Warn("fuse failed", zap.Error(err))
		return
	}

	if ctx.Err() != nil {
		return
	}

	rt.log.Info("cortex tick", zap.String("mode", rt.manager.CurrentMode()), zap.String("prompt", prompt))

	response, err := current.cortexLLM.Call(ctx, prompt, nil)
	if err != nil {
		rt.log.Warn("llm call failed", zap.Error(err))
		return
	}

	if len(response.ToolCalls) > 0 {
		calls, err := current.actionOrchestrator.ParseCalls(toolCallsToMaps(response.ToolCalls))
		if err != nil {
			rt.log.Warn("parse action calls failed", zap.Error(err))
		} else {
			for _, res := range current.actionOrchestrator.Submit(ctx, calls) {
				if res.Err != nil {
					rt.log.Warn("action failed",
						zap.String("action", res.ActionName),
						zap.Error(res.Err),
					)
				}
			}
		}
	}

	rt.ioProvider.RecordTick(tickStart)
}

// watchConfig polls the config file and logs a warning when it changes.
func (rt *Runtime) watchConfig(ctx context.Context) {
	if rt.systemConfig.Name == "" {
		return
	}
	path := fmt.Sprintf("config/%s.json5", rt.systemConfig.Name)
	info, err := os.Stat(path)
	if err != nil {
		return
	}
	lastMod := info.ModTime()
	interval := time.Duration(rt.opts.CheckInterval * float64(time.Second))
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			info, err := os.Stat(path)
			if err != nil {
				continue
			}
			if info.ModTime().After(lastMod) {
				lastMod = info.ModTime()
				rt.log.Info("config file changed — hot-reload not yet implemented; restart to apply changes",
					zap.String("path", path))
			}
		}
	}
}
