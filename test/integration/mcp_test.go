//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/mcp"
)

const mcpTestURLEnv = "OM1_TEST_MCP_URL"

func TestMain(m *testing.M) {
	server, _ := newWeatherServer()
	handler := mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server }, nil)
	httpServer := httptest.NewServer(handler)

	if err := os.Setenv(mcpTestURLEnv, httpServer.URL); err != nil {
		panic(err)
	}

	code := m.Run()

	httpServer.Close()
	os.Exit(code)
}

type schemaHolder struct {
	schemas []map[string]any
}

func (h *schemaHolder) SetSchemas(s []map[string]any)     { h.schemas = s }
func (h *schemaHolder) FunctionSchemas() []map[string]any { return h.schemas }

func newWeatherServer() (*mcpsdk.Server, *int64) {
	var calls int64

	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "test-weather", Version: "v1.0.0"}, nil)
	server.AddTool(
		&mcpsdk.Tool{
			Name:        "get_weather",
			Description: "Get the current weather for a city",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
		},
		func(_ context.Context, req *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			atomic.AddInt64(&calls, 1)
			var args struct {
				City string `json:"city"`
			}
			_ = json.Unmarshal(req.Params.Arguments, &args)
			return &mcpsdk.CallToolResult{
				Content: []mcpsdk.Content{
					&mcpsdk.TextContent{Text: "The weather in " + args.City + " is sunny, 22C."},
				},
			}, nil
		},
	)

	return server, &calls
}

func serveMCP(t *testing.T, server *mcpsdk.Server) string {
	t.Helper()
	handler := mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server }, nil)
	httpServer := httptest.NewServer(handler)
	t.Cleanup(httpServer.Close)
	return httpServer.URL
}

func TestMCPClientManagerEndToEnd(t *testing.T) {
	server, calls := newWeatherServer()
	url := serveMCP(t, server)

	manager := mcp.NewClientManager([]config.MCPSpec{
		{Name: "weather", Transport: "http", URL: url},
	}, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	require.NoError(t, manager.Start(ctx))
	t.Cleanup(func() { _ = manager.Stop() })

	const toolKey = "mcp_weather_get_weather"
	require.True(t, manager.IsMCPTool(toolKey))
	require.False(t, manager.IsMCPTool("get_weather"), "bare tool name is not registered")

	schemas := manager.ToolSchemas()
	require.Len(t, schemas, 1)
	fn := schemas[0]["function"].(map[string]any)
	require.Equal(t, toolKey, fn["name"])
	require.Equal(t, "Get the current weather for a city", fn["description"])

	descriptions := manager.ToolDescriptions()
	require.Contains(t, descriptions, "[MCP Tools]")
	require.Contains(t, descriptions, "- mcp_weather_get_weather(city: string)")

	result, err := manager.CallTool(ctx, toolKey, map[string]any{"city": "Paris"})
	require.NoError(t, err)
	require.Equal(t, "The weather in Paris is sunny, 22C.", result)
	require.Equal(t, int64(1), atomic.LoadInt64(calls), "the server-side tool ran exactly once")

	_, err = manager.CallTool(ctx, "mcp_weather_missing", nil)
	require.Error(t, err)
}

func TestMCPClientManagerHeaderInjection(t *testing.T) {
	server, _ := newWeatherServer()
	mcpHandler := mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server }, nil)

	var sawAuth atomic.Value
	sawAuth.Store("")
	wrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			sawAuth.Store(auth)
		}
		mcpHandler.ServeHTTP(w, r)
	})
	httpServer := httptest.NewServer(wrapped)
	t.Cleanup(httpServer.Close)

	manager := mcp.NewClientManager([]config.MCPSpec{
		{
			Name:      "weather",
			Transport: "http",
			URL:       httpServer.URL,
			Headers:   map[string]string{"Authorization": "Bearer test-token"},
		},
	}, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	require.NoError(t, manager.Start(ctx))
	t.Cleanup(func() { _ = manager.Stop() })

	require.Equal(t, "Bearer test-token", sawAuth.Load(),
		"configured Authorization header is injected on requests")
}

func TestMCPOrchestratorResolveEndToEnd(t *testing.T) {
	server, calls := newWeatherServer()
	url := serveMCP(t, server)

	manager := mcp.NewClientManager([]config.MCPSpec{
		{Name: "weather", Transport: "http", URL: url},
	}, zap.NewNop())
	orchestrator := mcp.NewOrchestrator(manager, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	holder := &schemaHolder{}
	require.NoError(t, orchestrator.Start(ctx, holder))
	t.Cleanup(func() { _ = orchestrator.Stop() })
	require.Len(t, holder.FunctionSchemas(), 1, "MCP tool schema injected into the LLM")

	callLLM := func(_ context.Context, prompt string) (*llm.Response, error) {
		require.Contains(t, prompt, "The weather in Tokyo is sunny")
		return &llm.Response{ToolCalls: []llm.ToolCall{
			{Name: "speak", Arguments: map[string]any{"text": "It's sunny in Tokyo!"}},
		}}, nil
	}

	initial := []llm.ToolCall{
		{Name: "mcp_weather_get_weather", Arguments: map[string]any{"city": "Tokyo"}},
	}
	final := orchestrator.Resolve(ctx, "base prompt", initial, callLLM, nil)

	require.Len(t, final, 1, "the MCP call is resolved and only the OM1 action remains")
	require.Equal(t, "speak", final[0].Name)
	require.Equal(t, "It's sunny in Tokyo!", final[0].Arguments["text"])
	require.Equal(t, int64(1), atomic.LoadInt64(calls), "the remote tool was invoked once")
}

func TestMCPRuntimeEndToEnd(t *testing.T) {
	server, calls := newWeatherServer()
	url := serveMCP(t, server)

	session, s := newSession()

	rules := []any{
		map[string]any{
			"when_contains": "[Tool Results]",
			"tool_calls": []any{
				map[string]any{
					"name":      "speak",
					"arguments": map[string]any{"text": "It is sunny in Berlin"},
				},
			},
		},
		map[string]any{
			"when_contains": "",
			"tool_calls": []any{
				map[string]any{
					"name":      "mcp_weather_get_weather",
					"arguments": map[string]any{"city": "Berlin"},
				},
			},
		},
	}

	mode := config.ModeConfig{
		Name:        "default",
		DisplayName: "default",
		Hertz:       50,
		AgentInputs: []config.PluginSpec{
			{Type: mockInputType, Config: map[string]any{"triggers_tick": true, "text": "what is the weather"}},
		},
		CortexLLM: config.PluginSpec{Type: scriptedLLMType, Config: map[string]any{"rules": rules}},
		AgentActions: []config.ActionSpec{
			{Name: "speak", LLMLabel: "speak", Connector: recordingConnKey},
		},
		MCPServers: []config.MCPSpec{
			{Name: "weather", Transport: "http", URL: url},
		},
	}

	cfg := &config.SystemConfig{
		Version:          "v1.0.0",
		Name:             "mcp_runtime_test",
		Hertz:            50,
		SystemPromptBase: "You are a helpful assistant.",
		DefaultMode:      "default",
		Modes:            map[string]config.ModeConfig{"default": mode},
	}

	recorded := runRuntime(cfg, session, s, 1, 5*time.Second)

	require.NotEmpty(t, recorded, "the runtime executed at least one OM1 action")
	require.True(t, matchesAny(expectedCall{
		Action:      "speak",
		ArgContains: map[string]string{"text": "sunny in Berlin"},
	}, recorded), "the follow-up OM1 action fired after the MCP tool resolved.\nrecorded: %s", describe(recorded))

	require.GreaterOrEqual(t, atomic.LoadInt64(calls), int64(1),
		"the real MCP weather tool was invoked by the runtime")
}
