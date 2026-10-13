package runtime

import (
	"encoding/json"
	"sort"

	"github.com/openmind/om1/internal/llm"
)

// repeatGuard suppresses a set of actions that is identical to the set already taken for the same
// observations. It protects against a model that re-acts on input it has already handled: the second
// identical response to an unchanged observation (including "no new observations") is dropped. Any change in
// the observations or in the requested actions resets the guard.
type repeatGuard struct {
	observations string
	actions      string
}

// Allow reports whether the actions may run for these observations and records them if so.
func (g *repeatGuard) Allow(observations string, calls []llm.ToolCall) bool {
	key := actionKey(calls)
	if key != "" && key == g.actions && observations == g.observations {
		return false
	}
	g.observations = observations
	g.actions = key
	return true
}

// actionKey is a canonical string for a set of tool calls, independent of their order.
func actionKey(calls []llm.ToolCall) string {
	if len(calls) == 0 {
		return ""
	}
	parts := make([]string, 0, len(calls))
	for _, c := range calls {
		args, _ := json.Marshal(c.Arguments)
		parts = append(parts, c.Name+"("+string(args)+")")
	}
	sort.Strings(parts)
	b, _ := json.Marshal(parts)
	return string(b)
}
