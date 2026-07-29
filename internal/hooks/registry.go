package hooks

import (
	"context"
	"fmt"
	"sync"
)

// HookFunc is the signature of a "function"-type lifecycle hook. It receives the
// Runner (for the logger and shared helpers), the execution context, the hook's
// handler_config map, and the template vars passed to Run.
//
// The signature matches a method expression on *Runner, e.g.
// (*Runner).greetingStartHook, so methods can be registered directly without a
// wrapping closure.
type HookFunc func(r *Runner, ctx context.Context, cfg, vars map[string]any) error

// hookRegistry maps "module.function" to the registered HookFunc implementation for that hook.
var hookRegistry = map[string]HookFunc{}

// RegisterHook registers a function hook implementation for the given module and function names.
// The module and function names are used in the hook's HandlerConfig to specify which implementation to run for "function"-type hooks.
func RegisterHook(module, function string, fn HookFunc) {
	key := module + "." + function
	if _, exists := hookRegistry[key]; exists {
		panic("hooks: duplicate function hook registration for " + key)
	}
	hookRegistry[key] = fn
}

// lookupHook returns the registered implementation for module.function.
func lookupHook(module, function string) (HookFunc, bool) {
	fn, ok := hookRegistry[module+"."+function]
	return fn, ok
}

// defaultRunner is the Runner used by Invoke for callers outside the
// lifecycle-hook flow, such as a background task reacting to a controller
// button. The runtime publishes the active mode's Runner here so those callers
// inherit its memory manager and logger.
var (
	defaultRunnerMu sync.RWMutex
	defaultRunner   *Runner
)

// SetDefaultRunner publishes the Runner that Invoke should use.
func SetDefaultRunner(r *Runner) {
	defaultRunnerMu.Lock()
	defaultRunner = r
	defaultRunnerMu.Unlock()
}

// Invoke runs a registered function hook directly, outside the lifecycle flow,
// letting a manual trigger reuse a hook that is otherwise reachable only from a
// mode transition.
func Invoke(ctx context.Context, module, function string, cfg, vars map[string]any) error {
	defaultRunnerMu.RLock()
	r := defaultRunner
	defaultRunnerMu.RUnlock()

	if r == nil {
		return fmt.Errorf("hooks: no default runner registered for %s.%s", module, function)
	}

	handler, ok := lookupHook(module, function)
	if !ok {
		return fmt.Errorf("hooks: unknown function hook %s.%s", module, function)
	}

	return handler(r, ctx, cfg, vars)
}
