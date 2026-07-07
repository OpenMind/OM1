package mcp

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
)

func configMCP(transport, command, url string) config.MCPSpec {
	return config.MCPSpec{Name: "test", Transport: transport, Command: command, URL: url}
}

func TestToolSchema(t *testing.T) {
	tool := Tool{
		Key:         "mcp_weather_get",
		Description: "Get the weather",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
	}

	schema := tool.Schema()
	require.Equal(t, "function", schema["type"])

	fn := schema["function"].(map[string]any)
	require.Equal(t, "mcp_weather_get", fn["name"])
	require.Equal(t, "Get the weather", fn["description"])
	require.Equal(t, tool.InputSchema, fn["parameters"])
}

func TestToolDescribe(t *testing.T) {
	tool := Tool{
		Key:         "mcp_weather_get",
		Description: "Get the weather",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
				"days": map[string]any{"type": "integer"},
			},
		},
	}

	require.Equal(t, "- mcp_weather_get(city: string, days: integer): Get the weather", tool.describe())
}

func TestNormalizeSchema(t *testing.T) {
	empty := map[string]any{"type": "object", "properties": map[string]any{}}

	require.Equal(t, empty, normalizeSchema(nil))

	got := normalizeSchema(struct {
		Type string `json:"type"`
	}{Type: "object"})
	require.Equal(t, "object", got["type"])

	in := map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{}}}
	require.Equal(t, in, normalizeSchema(in))
}

func TestClientManagerRegistryHelpers(t *testing.T) {
	m := NewClientManager(nil, zap.NewNop())
	m.tools = map[string]Tool{
		"mcp_a_one": {Key: "mcp_a_one", ServerName: "a", OriginalName: "one", Description: "first",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}},
		"mcp_a_two": {Key: "mcp_a_two", ServerName: "a", OriginalName: "two", Description: "second",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}},
	}
	m.order = []string{"mcp_a_one", "mcp_a_two"}

	require.True(t, m.IsMCPTool("mcp_a_one"))
	require.False(t, m.IsMCPTool("some_om1_action"))

	schemas := m.ToolSchemas()
	require.Len(t, schemas, 2)
	require.Equal(t, "mcp_a_one", schemas[0]["function"].(map[string]any)["name"])

	descriptions := m.ToolDescriptions()
	require.True(t, strings.HasPrefix(descriptions, "[MCP Tools]"))
	require.Contains(t, descriptions, "- mcp_a_one(): first")
	require.Contains(t, descriptions, "- mcp_a_two(): second")
}

func TestClientManagerToolDescriptionsEmpty(t *testing.T) {
	m := NewClientManager(nil, zap.NewNop())
	require.Equal(t, "", m.ToolDescriptions())
	require.Empty(t, m.ToolSchemas())
}

func TestNewTransport(t *testing.T) {
	_, err := newTransport(configMCP("stdio", "", ""))
	require.Error(t, err, "stdio without command is rejected")

	_, err = newTransport(configMCP("sse", "", ""))
	require.Error(t, err, "sse without url is rejected")

	_, err = newTransport(configMCP("bogus", "echo", "http://x"))
	require.Error(t, err, "unknown transport is rejected")

	stdio, err := newTransport(configMCP("", "echo", ""))
	require.NoError(t, err, "empty transport defaults to stdio")
	require.NotNil(t, stdio)

	http, err := newTransport(configMCP("http", "", "http://localhost:9"))
	require.NoError(t, err)
	require.NotNil(t, http)
}
