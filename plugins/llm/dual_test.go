package llm

import (
	"context"
	"errors"
	"net/http"
	"testing"
	"time"

	"github.com/openmind/om1/internal/llm"
	"github.com/stretchr/testify/require"
)

type stubLLM struct {
	resp    *llm.Response
	err     error
	delay   time.Duration
	schemas []map[string]any
}

func (s *stubLLM) Call(ctx context.Context, _ string, _ []llm.Message) (*llm.Response, error) {
	if s.delay > 0 {
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return s.resp, s.err
}

func (s *stubLLM) SetSchemas(sc []map[string]any)    { s.schemas = sc }
func (s *stubLLM) FunctionSchemas() []map[string]any { return s.schemas }

func respWithCalls(names ...string) *llm.Response {
	r := &llm.Response{}
	for _, n := range names {
		r.ToolCalls = append(r.ToolCalls, llm.ToolCall{Name: n})
	}
	return r
}

func TestDualCallBothInTimeLocalHasCalls(t *testing.T) {
	d := &dualLLM{
		local: &stubLLM{resp: respWithCalls("speak")},
		cloud: &stubLLM{resp: &llm.Response{}},
	}
	resp, err := d.Call(context.Background(), "INPUT Voice: hi there", nil)
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "speak", resp.ToolCalls[0].Name)
}

func TestDualCallBothInTimeCloudHasCalls(t *testing.T) {
	d := &dualLLM{
		local: &stubLLM{resp: &llm.Response{}},
		cloud: &stubLLM{resp: respWithCalls("move")},
	}
	resp, err := d.Call(context.Background(), "Voice: move", nil)
	require.NoError(t, err)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "move", resp.ToolCalls[0].Name)
}

func TestDualCallBothInTimeEvalPicksCloud(t *testing.T) {
	srv, _ := captureServer(t, http.StatusOK, `{"choices":[{"message":{"content":"B"}}]}`)
	d := &dualLLM{
		local: &stubLLM{resp: respWithCalls("speak")},
		cloud: &stubLLM{resp: respWithCalls("move")},
		eval:  &openAICompatLLM{provider: "eval", config: compatConfig{BaseURL: srv.URL, APIKey: "k", Model: "m"}},
	}
	resp, err := d.Call(context.Background(), "Voice: do something", nil)
	require.NoError(t, err)
	require.Equal(t, "move", resp.ToolCalls[0].Name)
}

func TestDualCallBothFailReturnsEmpty(t *testing.T) {
	d := &dualLLM{
		local: &stubLLM{err: errors.New("local down")},
		cloud: &stubLLM{err: errors.New("cloud down")},
	}
	resp, err := d.Call(context.Background(), "prompt", nil)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.Empty(t, resp.ToolCalls)
}

func TestDualSelectBest(t *testing.T) {
	d := &dualLLM{}
	local := respWithCalls("speak")
	cloud := respWithCalls("move")
	empty := &llm.Response{}

	require.Same(t, local, d.selectBest(context.Background(), local, empty, ""), "only local has calls")
	require.Same(t, cloud, d.selectBest(context.Background(), empty, cloud, ""), "only cloud has calls")
	require.Same(t, empty, d.selectBest(context.Background(), empty, empty, ""), "neither has calls → local")
}

func TestDualEvaluateQuality(t *testing.T) {
	local := respWithCalls("speak")
	cloud := respWithCalls("move")

	cases := []struct {
		name     string
		response string
		want     string
	}{
		{"picks local on A", `{"choices":[{"message":{"content":"A"}}]}`, "local"},
		{"picks cloud on B", `{"choices":[{"message":{"content":"B"}}]}`, "cloud"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := captureServer(t, http.StatusOK, tc.response)
			d := &dualLLM{eval: &openAICompatLLM{config: compatConfig{BaseURL: srv.URL, APIKey: "k", Model: "m"}}}
			require.Equal(t, tc.want, d.evaluateQuality(context.Background(), local, cloud, "ctx"))
		})
	}
}

func TestDualEvaluateQualityDefaultsToLocalOnError(t *testing.T) {
	srv, _ := captureServer(t, http.StatusInternalServerError, `boom`)
	d := &dualLLM{eval: &openAICompatLLM{config: compatConfig{BaseURL: srv.URL, APIKey: "k", Model: "m"}}}
	require.Equal(t, "local", d.evaluateQuality(context.Background(), respWithCalls("a"), respWithCalls("b"), "ctx"))
}

func TestToolCallsToEval(t *testing.T) {
	got := toolCallsToEval([]llm.ToolCall{{Name: "speak", Arguments: map[string]any{"text": "hi"}}})
	require.Len(t, got, 1)
	require.Equal(t, "speak", got[0]["type"])
	require.Equal(t, map[string]any{"text": "hi"}, got[0]["value"])
}

func TestFirstUsable(t *testing.T) {
	usable := respWithCalls("speak")
	require.Nil(t, firstUsable([]dualResult{{resp: nil}, {resp: nil}}))
	require.Same(t, usable, firstUsable([]dualResult{{resp: nil}, {resp: usable}}))
}
