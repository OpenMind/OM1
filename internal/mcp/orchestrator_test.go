package mcp

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/llm"
)

type fakeClient struct {
	mu        sync.Mutex
	mcpTools  map[string]bool
	schemas   []map[string]any
	responses map[string]string
	errs      map[string]error
	calls     []string
}

func newFakeClient(tools ...string) *fakeClient {
	set := make(map[string]bool)
	for _, t := range tools {
		set[t] = true
	}
	return &fakeClient{
		mcpTools:  set,
		responses: make(map[string]string),
		errs:      make(map[string]error),
	}
}

func (f *fakeClient) Start(context.Context) error   { return nil }
func (f *fakeClient) Stop() error                   { return nil }
func (f *fakeClient) ToolSchemas() []map[string]any { return f.schemas }
func (f *fakeClient) IsMCPTool(name string) bool    { return f.mcpTools[name] }

func (f *fakeClient) CallTool(_ context.Context, key string, args map[string]any) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, fmt.Sprintf("%s%v", key, args))
	if err := f.errs[key]; err != nil {
		return "", err
	}
	return f.responses[key], nil
}

func (f *fakeClient) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

type fakeSchemaHolder struct {
	schemas []map[string]any
}

func (f *fakeSchemaHolder) SetSchemas(s []map[string]any)     { f.schemas = s }
func (f *fakeSchemaHolder) FunctionSchemas() []map[string]any { return f.schemas }

func fnSchema(name string) map[string]any {
	return map[string]any{"type": "function", "function": map[string]any{"name": name}}
}

func TestOrchestratorStartInjectsSchemas(t *testing.T) {
	client := newFakeClient("mcp_srv_tool")
	client.schemas = []map[string]any{fnSchema("mcp_srv_tool")}

	holder := &fakeSchemaHolder{schemas: []map[string]any{
		fnSchema("move"),
		fnSchema("mcp_stale_old"),
	}}

	o := NewOrchestrator(client, zap.NewNop())
	require.NoError(t, o.Start(context.Background(), holder))

	names := schemaNames(holder.schemas)
	require.Equal(t, []string{"move", "mcp_srv_tool"}, names,
		"base schemas are preserved, stale MCP schemas dropped, fresh MCP schemas appended")
}

func TestOrchestratorResolveExecutesMCPThenReturnsOM1(t *testing.T) {
	client := newFakeClient("mcp_weather_get")
	client.responses["mcp_weather_get"] = "sunny"

	o := NewOrchestrator(client, zap.NewNop())

	var recallPrompts []string
	callLLM := func(_ context.Context, prompt string) (*llm.Response, error) {
		recallPrompts = append(recallPrompts, prompt)
		return &llm.Response{ToolCalls: []llm.ToolCall{{Name: "speak", Arguments: map[string]any{"text": "It is sunny"}}}}, nil
	}

	initial := []llm.ToolCall{{Name: "mcp_weather_get", Arguments: map[string]any{"city": "SF"}}}
	final := o.Resolve(context.Background(), "orig", initial, callLLM, nil)

	require.Len(t, final, 1)
	require.Equal(t, "speak", final[0].Name)
	require.Equal(t, 1, client.callCount(), "MCP tool executed once")
	require.Len(t, recallPrompts, 1)
	require.Contains(t, recallPrompts[0], "[Tool Results]")
	require.Contains(t, recallPrompts[0], "[mcp_weather_get] OK: sunny")
}

func TestOrchestratorResolveDeduplicatesAcrossRounds(t *testing.T) {
	client := newFakeClient("mcp_weather_get")
	client.responses["mcp_weather_get"] = "sunny"

	o := NewOrchestrator(client, zap.NewNop())

	callLLM := func(_ context.Context, _ string) (*llm.Response, error) {
		return &llm.Response{ToolCalls: []llm.ToolCall{
			{Name: "mcp_weather_get", Arguments: map[string]any{"city": "SF"}},
		}}, nil
	}

	initial := []llm.ToolCall{{Name: "mcp_weather_get", Arguments: map[string]any{"city": "SF"}}}
	final := o.Resolve(context.Background(), "orig", initial, callLLM, nil)

	require.Empty(t, final, "no OM1 actions remain")
	require.Equal(t, 1, client.callCount(), "an already-succeeded call is not repeated")
}

func TestOrchestratorResolveExecutesIntermediateOM1Actions(t *testing.T) {
	client := newFakeClient("mcp_weather_get")
	client.responses["mcp_weather_get"] = "sunny"

	o := NewOrchestrator(client, zap.NewNop())

	var executed [][]llm.ToolCall
	execOM1 := func(_ context.Context, calls []llm.ToolCall) {
		executed = append(executed, calls)
	}
	callLLM := func(_ context.Context, _ string) (*llm.Response, error) {
		return &llm.Response{}, nil
	}

	initial := []llm.ToolCall{
		{Name: "mcp_weather_get", Arguments: map[string]any{"city": "SF"}},
		{Name: "move", Arguments: map[string]any{"dir": "left"}},
	}
	final := o.Resolve(context.Background(), "orig", initial, callLLM, execOM1)

	require.Empty(t, final)
	require.Len(t, executed, 1, "intermediate OM1 actions executed once")
	require.Equal(t, "move", executed[0][0].Name)
}

func TestOrchestratorResolveDoesNotReexecuteOM1OnRecallFailure(t *testing.T) {
	client := newFakeClient("mcp_weather_get")
	client.responses["mcp_weather_get"] = "sunny"

	o := NewOrchestrator(client, zap.NewNop())

	var executed [][]llm.ToolCall
	execOM1 := func(_ context.Context, calls []llm.ToolCall) {
		executed = append(executed, calls)
	}
	// The recall call fails after the first round's actions have been dispatched.
	callLLM := func(_ context.Context, _ string) (*llm.Response, error) {
		return nil, fmt.Errorf("llm unavailable")
	}

	initial := []llm.ToolCall{
		{Name: "mcp_weather_get", Arguments: map[string]any{"city": "SF"}},
		{Name: "move", Arguments: map[string]any{"dir": "left"}},
	}
	final := o.Resolve(context.Background(), "orig", initial, callLLM, execOM1)

	require.Len(t, executed, 1, "intermediate OM1 actions dispatched once")
	require.Equal(t, "move", executed[0][0].Name)
	require.Empty(t, final, "already-dispatched OM1 actions must not be returned again on the error path")
}

func TestOrchestratorResolveNoMCPActions(t *testing.T) {
	client := newFakeClient("mcp_weather_get")
	o := NewOrchestrator(client, zap.NewNop())

	llmCalled := false
	callLLM := func(_ context.Context, _ string) (*llm.Response, error) {
		llmCalled = true
		return &llm.Response{}, nil
	}

	initial := []llm.ToolCall{{Name: "move", Arguments: map[string]any{"dir": "left"}}}
	final := o.Resolve(context.Background(), "orig", initial, callLLM, nil)

	require.Len(t, final, 1)
	require.Equal(t, "move", final[0].Name)
	require.False(t, llmCalled, "no recall when there are no MCP tools to run")
}

func TestExecuteSingleTreatsJSONErrorAsFailure(t *testing.T) {
	client := newFakeClient("mcp_srv_tool")
	client.responses["mcp_srv_tool"] = `{"error": "boom"}`

	o := NewOrchestrator(client, zap.NewNop())
	result := o.executeSingle(context.Background(), llm.ToolCall{Name: "mcp_srv_tool"})

	require.False(t, result.Success, "a JSON payload with an error field is a failure")
	require.Contains(t, result.Content, "boom")
}

func TestExecuteSingleError(t *testing.T) {
	client := newFakeClient("mcp_srv_tool")
	client.errs["mcp_srv_tool"] = fmt.Errorf("connection lost")

	o := NewOrchestrator(client, zap.NewNop())
	result := o.executeSingle(context.Background(), llm.ToolCall{Name: "mcp_srv_tool"})

	require.False(t, result.Success)
	require.Contains(t, result.Content, "connection lost")
}

func TestCallSignatureStableAcrossArgOrder(t *testing.T) {
	o := NewOrchestrator(newFakeClient(), zap.NewNop())
	a := llm.ToolCall{Name: "t", Arguments: map[string]any{"a": 1, "b": 2}}
	b := llm.ToolCall{Name: "t", Arguments: map[string]any{"b": 2, "a": 1}}
	require.Equal(t, o.callSignature(a), o.callSignature(b),
		"signatures are stable regardless of map iteration order")
}

func schemaNames(schemas []map[string]any) []string {
	var names []string
	for _, s := range schemas {
		fn, _ := s["function"].(map[string]any)
		if name, ok := fn["name"].(string); ok {
			names = append(names, name)
		}
	}
	return names
}
