package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/config"
	"github.com/openmind/om1/internal/logger"
)

const (
	TransportStdio = "stdio"
	TransportSSE   = "sse"
	TransportHTTP  = "http"
)

const (
	clientName    = "om1"
	clientVersion = "0.1.0"
)

// Tool holds metadata for a single tool exposed by an MCP server.
type Tool struct {
	// Key is the LLM-facing name, formatted as "mcp_{server}_{tool}".
	Key string
	// ServerName is the configured name of the owning MCP server.
	ServerName string
	// OriginalName is the tool name as reported by the server.
	OriginalName string
	// Description is a human-readable description of the tool.
	Description string
	// InputSchema is the JSON Schema for the tool's parameters.
	InputSchema map[string]any
}

// Schema converts the tool to an OpenAI function-calling schema.
func (t Tool) Schema() map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        t.Key,
			"description": t.Description,
			"parameters":  t.InputSchema,
		},
	}
}

// describe returns a one-line description of the tool for LLM prompts.
func (t Tool) describe() string {
	var params []string
	if props, ok := t.InputSchema["properties"].(map[string]any); ok {
		names := make([]string, 0, len(props))
		for name := range props {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			typeName := "any"
			if info, ok := props[name].(map[string]any); ok {
				if typ, ok := info["type"].(string); ok {
					typeName = typ
				}
			}
			params = append(params, fmt.Sprintf("%s: %s", name, typeName))
		}
	}
	return fmt.Sprintf("- %s(%s): %s", t.Key, strings.Join(params, ", "), t.Description)
}

// ClientManager manages connections to multiple MCP servers and their tools.
type ClientManager struct {
	configs []config.MCPSpec
	log     *zap.Logger

	mu       sync.RWMutex
	sessions map[string]*mcpsdk.ClientSession
	tools    map[string]Tool
	order    []string
	started  bool
}

// NewClientManager creates a ClientManager for the given server configurations.
func NewClientManager(configs []config.MCPSpec, log *zap.Logger) *ClientManager {
	if log == nil {
		log = logger.Get()
	}
	return &ClientManager{
		configs:  configs,
		log:      log,
		sessions: make(map[string]*mcpsdk.ClientSession),
		tools:    make(map[string]Tool),
	}
}

// Start connects to all configured MCP servers and registers their tools.
func (m *ClientManager) Start(ctx context.Context) error {
	m.mu.Lock()
	if m.started {
		m.mu.Unlock()
		return nil
	}
	m.mu.Unlock()

	var wg sync.WaitGroup
	for _, cfg := range m.configs {
		wg.Add(1)
		go func(cfg config.MCPSpec) {
			defer wg.Done()
			if err := m.connect(ctx, cfg); err != nil {
				m.log.Error("MCP server connection failed",
					zap.String("server", cfg.Name), zap.Error(err))
			}
		}(cfg)
	}
	wg.Wait()

	m.mu.Lock()
	m.started = true
	m.mu.Unlock()
	return nil
}

// connect establishes a session with a single MCP server and registers its tools.
func (m *ClientManager) connect(ctx context.Context, cfg config.MCPSpec) error {
	transport, err := newTransport(cfg)
	if err != nil {
		return err
	}

	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: clientName, Version: clientVersion}, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return fmt.Errorf("connect: %w", err)
	}

	listCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	result, err := session.ListTools(listCtx, nil)
	if err != nil {
		_ = session.Close()
		return fmt.Errorf("list tools: %w", err)
	}

	m.mu.Lock()
	m.sessions[cfg.Name] = session
	names := make([]string, 0, len(result.Tools))
	for _, tool := range result.Tools {
		key := fmt.Sprintf("mcp_%s_%s", cfg.Name, tool.Name)
		description := tool.Description
		if description == "" {
			description = "MCP tool: " + tool.Name
		}
		m.tools[key] = Tool{
			Key:          key,
			ServerName:   cfg.Name,
			OriginalName: tool.Name,
			Description:  description,
			InputSchema:  normalizeSchema(tool.InputSchema),
		}
		m.order = append(m.order, key)
		names = append(names, tool.Name)
	}
	m.mu.Unlock()

	m.log.Info("MCP server connected",
		zap.String("server", cfg.Name),
		zap.Int("tools", len(result.Tools)),
		zap.Strings("names", names))
	return nil
}

// newTransport builds the SDK transport for a server based on its configured
// transport type.
func newTransport(cfg config.MCPSpec) (mcpsdk.Transport, error) {
	transport := cfg.Transport
	if transport == "" {
		transport = TransportStdio
	}

	switch transport {
	case TransportStdio:
		if cfg.Command == "" {
			return nil, fmt.Errorf("MCP server %q requires 'command' for stdio transport", cfg.Name)
		}
		cmd := exec.Command(cfg.Command, cfg.Args...)
		cmd.Env = mergeEnv(cfg.Env)
		return &mcpsdk.CommandTransport{Command: cmd}, nil

	case TransportSSE:
		if cfg.URL == "" {
			return nil, fmt.Errorf("MCP server %q requires 'url' for sse transport", cfg.Name)
		}
		return &mcpsdk.SSEClientTransport{Endpoint: cfg.URL, HTTPClient: httpClient(cfg.Headers)}, nil

	case TransportHTTP:
		if cfg.URL == "" {
			return nil, fmt.Errorf("MCP server %q requires 'url' for http transport", cfg.Name)
		}
		return &mcpsdk.StreamableClientTransport{Endpoint: cfg.URL, HTTPClient: httpClient(cfg.Headers)}, nil

	default:
		return nil, fmt.Errorf("MCP server %q: unsupported transport %q", cfg.Name, transport)
	}
}

// Stop closes all MCP server sessions and clears the tool registry.
func (m *ClientManager) Stop() error {
	m.mu.Lock()
	sessions := m.sessions
	m.sessions = make(map[string]*mcpsdk.ClientSession)
	m.tools = make(map[string]Tool)
	m.order = nil
	m.started = false
	m.mu.Unlock()

	for name, session := range sessions {
		if err := session.Close(); err != nil {
			m.log.Warn("error closing MCP connection", zap.String("server", name), zap.Error(err))
		}
	}
	return nil
}

// ToolSchemas returns OpenAI-format function schemas for all registered tools.
func (m *ClientManager) ToolSchemas() []map[string]any {
	m.mu.RLock()
	defer m.mu.RUnlock()

	schemas := make([]map[string]any, 0, len(m.order))
	for _, key := range m.order {
		if tool, ok := m.tools[key]; ok {
			schemas = append(schemas, tool.Schema())
		}
	}
	return schemas
}

// ToolDescriptions returns a prompt block describing the registered MCP tools,
// or an empty string if there are none.
func (m *ClientManager) ToolDescriptions() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.order) == 0 {
		return ""
	}

	var builder strings.Builder
	builder.WriteString("[MCP Tools]\n")
	builder.WriteString("Use these tools to get external information. ")
	builder.WriteString("Call the tool first, then use the result to respond.\n")
	for i, key := range m.order {
		tool, ok := m.tools[key]
		if !ok {
			continue
		}
		if i > 0 {
			builder.WriteString("\n")
		}
		builder.WriteString(tool.describe())
	}
	return builder.String()
}

// IsMCPTool reports whether the given tool name belongs to an MCP server.
func (m *ClientManager) IsMCPTool(name string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.tools[name]
	return ok
}

// CallTool invokes an MCP tool by key and returns its text result.
func (m *ClientManager) CallTool(ctx context.Context, key string, arguments map[string]any) (string, error) {
	m.mu.RLock()
	tool, ok := m.tools[key]
	var session *mcpsdk.ClientSession
	if ok {
		session = m.sessions[tool.ServerName]
	}
	m.mu.RUnlock()

	if !ok {
		return "", fmt.Errorf("unknown MCP tool: %s", key)
	}
	if session == nil {
		return "", fmt.Errorf("MCP server %q not connected", tool.ServerName)
	}

	result, err := session.CallTool(ctx, &mcpsdk.CallToolParams{
		Name:      tool.OriginalName,
		Arguments: arguments,
	})
	if err != nil {
		return "", err
	}

	var texts []string
	for _, content := range result.Content {
		if text, ok := content.(*mcpsdk.TextContent); ok {
			texts = append(texts, text.Text)
		}
	}
	if len(texts) > 0 {
		return strings.Join(texts, "\n"), nil
	}
	return fmt.Sprintf("%v", result.Content), nil
}

// mergeEnv builds the environment for a stdio subprocess by appending the
// configured key/value pairs to the current process environment.
func mergeEnv(env map[string]string) []string {
	if len(env) == 0 {
		return nil // nil means inherit the parent environment
	}
	merged := append([]string{}, os.Environ()...)
	for k, v := range env {
		merged = append(merged, k+"="+v)
	}
	return merged
}

// httpClient returns an *http.Client that injects the given headers on every
// request, or the default client when there are no headers.
func httpClient(headers map[string]string) *http.Client {
	if len(headers) == 0 {
		return nil
	}
	return &http.Client{
		Timeout:   300 * time.Second,
		Transport: &headerRoundTripper{headers: headers, base: http.DefaultTransport},
	}
}

// headerRoundTripper adds a fixed set of headers to every outgoing request.
type headerRoundTripper struct {
	headers map[string]string
	base    http.RoundTripper
}

// RoundTrip implements http.RoundTripper.
func (h *headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	for k, v := range h.headers {
		clone.Header.Set(k, v)
	}
	return h.base.RoundTrip(clone)
}

// normalizeSchema converts a JSON-marshalable value to a map[string]any.
func normalizeSchema(schema any) map[string]any {
	empty := map[string]any{"type": "object", "properties": map[string]any{}}
	if schema == nil {
		return empty
	}
	if m, ok := schema.(map[string]any); ok {
		return m
	}
	data, err := json.Marshal(schema)
	if err != nil {
		return empty
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil || m == nil {
		return empty
	}
	return m
}
