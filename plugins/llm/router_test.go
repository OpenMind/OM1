package llm

import (
	"context"
	"regexp"
	"testing"

	"github.com/openmind/om1/internal/llm"
	"github.com/stretchr/testify/require"
)

func newTestRouter() (*routerLLM, *stubLLM, *stubLLM) {
	chat := &stubLLM{resp: respWithCalls("speak")}
	command := &stubLLM{resp: respWithCalls("move")}
	chatRoute := &route{name: "chat", llm: chat, regexes: []*regexp.Regexp{regexp.MustCompile(`(?i)\?`)}}
	cmdRoute := &route{
		name:     "command",
		llm:      command,
		keywords: []string{"sit", "come", "fetch", "spin"},
	}
	r := &routerLLM{routes: []*route{cmdRoute, chatRoute}, def: chatRoute}
	return r, chat, command
}

func TestRouterPicksCommandOnKeyword(t *testing.T) {
	r, _, _ := newTestRouter()
	require.Equal(t, "command", r.pick("INPUT Voice: come here and sit").name)
}

func TestRouterPicksChatOnQuestion(t *testing.T) {
	r, _, _ := newTestRouter()
	require.Equal(t, "chat", r.pick("Voice: how are you today?").name)
}

func TestRouterFallsBackToDefault(t *testing.T) {
	r, _, _ := newTestRouter()
	require.Equal(t, "chat", r.pick("Voice: the weather is nice").name, "no match → default route")
}

func TestRouterEmptyVoiceUsesDefault(t *testing.T) {
	r, _, _ := newTestRouter()
	require.Equal(t, "chat", r.pick("no voice line here").name)
}

func TestRouterHighestScoreWins(t *testing.T) {
	r, _, _ := newTestRouter()
	// Two command keywords beat a single chat question mark.
	require.Equal(t, "command", r.pick("Voice: sit and fetch, ok?").name)
}

func TestRouterCallDispatchesToChosen(t *testing.T) {
	r, _, command := newTestRouter()
	resp, err := r.Call(context.Background(), "Voice: fetch the ball", nil)
	require.NoError(t, err)
	require.Same(t, command.resp, resp)
}

func TestRouterSetSchemasPropagates(t *testing.T) {
	r, chat, command := newTestRouter()
	schemas := []map[string]any{{"name": "speak"}}
	r.SetSchemas(schemas)
	require.Equal(t, schemas, chat.schemas)
	require.Equal(t, schemas, command.schemas)
}

func TestNewRouterRequiresRoutes(t *testing.T) {
	_, err := NewRouter(map[string]any{})
	require.Error(t, err)
}

func TestNewRouterUnknownDefaultRoute(t *testing.T) {
	_, err := NewRouter(map[string]any{
		"default_route": "nope",
		"routes": []map[string]any{
			{"name": "chat", "llm_type": "GeminiLLM", "llm_config": map[string]any{"api_key": "k"}},
		},
	})
	require.Error(t, err)
}

func TestNewRouterBuildsRoutes(t *testing.T) {
	got, err := NewRouter(map[string]any{
		"api_key":       "shared-key",
		"default_route": "chat",
		"routes": []map[string]any{
			{
				"name":     "command",
				"llm_type": "GeminiLLM",
				"keywords": []string{"sit", "come"},
				"patterns": []string{`\bstop\b`},
			},
			{
				"name":       "chat",
				"llm_type":   "OpenRouter",
				"llm_config": map[string]any{"temperature": 0.9},
			},
		},
	})
	require.NoError(t, err)
	rt := got.(*routerLLM)
	require.Len(t, rt.routes, 2)
	require.Equal(t, "chat", rt.def.name)
	require.Equal(t, "command", rt.pick("Voice: STOP now").name)
}

var _ llm.LLM = (*routerLLM)(nil)
