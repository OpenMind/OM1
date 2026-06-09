package hooks

import "context"

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
