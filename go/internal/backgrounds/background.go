package backgrounds

import "context"

// Background is the interface every background plugin must implement.
//
// The orchestrator calls Run repeatedly in a loop — one call per iteration —
// mirroring the Python model where _run_background_loop calls background.run()
// over and over.  Run should therefore do one unit of work and return.  Use
//
//	select { case <-ctx.Done(): return; case <-time.After(d): }
//
// for any sleeping so that mode transitions cancel the sleep immediately.
//
// Stop is called once by the orchestrator after the loop exits (ctx cancelled),
// and is the place to close connections or release resources.
type Background interface {
	Run(ctx context.Context)
	Stop()
}

type Factory func(cfg map[string]any) (Background, error)

var registry = map[string]Factory{}

func Register(typeName string, f Factory) {
	registry[typeName] = f
}

func Load(typeName string, cfg map[string]any) (Background, error) {
	f, ok := registry[typeName]
	if !ok {
		return nil, &UnknownPluginError{Name: typeName}
	}
	return f(cfg)
}

type UnknownPluginError struct{ Name string }

func (e *UnknownPluginError) Error() string { return "background plugin not found: " + e.Name }
