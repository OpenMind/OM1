package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/llm"
	"github.com/openmind/om1/internal/logger"
)

const (
	defaultMaxConcurrency = 5
	defaultMaxRounds      = 5
	defaultToolTimeout    = 10 * time.Second
)

type Client interface {
	Start(ctx context.Context) error
	Stop() error
	ToolSchemas() []map[string]any
	IsMCPTool(name string) bool
	CallTool(ctx context.Context, key string, arguments map[string]any) (string, error)
}

type SchemaHolder interface {
	SetSchemas(schemas []map[string]any)
	FunctionSchemas() []map[string]any
}

type LLMCaller func(ctx context.Context, prompt string) (*llm.Response, error)

type OM1Executor func(ctx context.Context, calls []llm.ToolCall)

type ToolResult struct {
	ToolKey string
	Success bool
	Content string
}

type Orchestrator struct {
	client         Client
	log            *zap.Logger
	maxConcurrency int
	maxRounds      int
	toolTimeout    time.Duration
}

// NewOrchestrator creates an Orchestrator wrapping the given client.
func NewOrchestrator(client Client, log *zap.Logger) *Orchestrator {
	if log == nil {
		log = logger.Get()
	}
	return &Orchestrator{
		client:         client,
		log:            log,
		maxConcurrency: defaultMaxConcurrency,
		maxRounds:      defaultMaxRounds,
		toolTimeout:    defaultToolTimeout,
	}
}

// Start connects the MCP servers and injects their tool schemas into the LLM,
// preserving any existing non-MCP (base) schemas.
func (o *Orchestrator) Start(ctx context.Context, llm SchemaHolder) error {
	if err := o.client.Start(ctx); err != nil {
		return err
	}

	mcpSchemas := o.client.ToolSchemas()
	base := make([]map[string]any, 0, len(llm.FunctionSchemas()))
	for _, schema := range llm.FunctionSchemas() {
		if !isMCPSchema(schema) {
			base = append(base, schema)
		}
	}
	llm.SetSchemas(append(base, mcpSchemas...))

	o.log.Info("MCP orchestrator started",
		zap.Int("mcp_tools", len(mcpSchemas)),
		zap.Int("base_tools", len(base)))
	return nil
}

// Stop disconnects the MCP servers.
func (o *Orchestrator) Stop() error {
	return o.client.Stop()
}

// Resolve executes multiple rounds of MCP tool calls, returning any remaining OM1 actions.
func (o *Orchestrator) Resolve(
	ctx context.Context,
	prompt string,
	calls []llm.ToolCall,
	callLLM LLMCaller,
	execOM1 OM1Executor,
) []llm.ToolCall {
	succeeded := make(map[string]bool)
	current := calls

	for round := 0; round < o.maxRounds; round++ {
		mcpActions := o.pendingMCPActions(current, succeeded)
		if len(mcpActions) == 0 {
			break
		}

		results := o.executeActions(ctx, mcpActions)
		for i, action := range mcpActions {
			if results[i].Success {
				succeeded[o.callSignature(action)] = true
			}
		}

		if om1 := o.extractOM1Actions(current); len(om1) > 0 && execOM1 != nil {
			execOM1(ctx, om1)
			current = o.stripOM1(current)
		}

		o.log.Info("MCP round",
			zap.Int("round", round+1),
			zap.Int("max_rounds", o.maxRounds),
			zap.Int("tools", len(mcpActions)))

		recallPrompt := o.buildResultPrompt(prompt, results)
		response, err := callLLM(ctx, recallPrompt)
		if err != nil {
			o.log.Warn("MCP recall LLM call failed", zap.Error(err))
			break
		}
		if response == nil {
			break
		}
		current = response.ToolCalls
	}

	return o.extractOM1Actions(current)
}

// pendingMCPActions returns MCP tool calls that have not yet succeeded.
func (o *Orchestrator) pendingMCPActions(calls []llm.ToolCall, succeeded map[string]bool) []llm.ToolCall {
	var pending []llm.ToolCall
	for _, call := range calls {
		if o.client.IsMCPTool(call.Name) && !succeeded[o.callSignature(call)] {
			pending = append(pending, call)
		}
	}
	return pending
}

// extractOM1Actions returns the non-MCP tool calls from calls.
func (o *Orchestrator) extractOM1Actions(calls []llm.ToolCall) []llm.ToolCall {
	var om1 []llm.ToolCall
	for _, call := range calls {
		if !o.client.IsMCPTool(call.Name) {
			om1 = append(om1, call)
		}
	}
	return om1
}

// stripOM1 removes non-MCP tool calls from calls, returning only MCP tool calls.
func (o *Orchestrator) stripOM1(calls []llm.ToolCall) []llm.ToolCall {
	var kept []llm.ToolCall
	for _, call := range calls {
		if o.client.IsMCPTool(call.Name) {
			kept = append(kept, call)
		}
	}
	return kept
}

// executeActions runs multiple MCP tool calls concurrently, respecting maxConcurrency.
func (o *Orchestrator) executeActions(ctx context.Context, actions []llm.ToolCall) []ToolResult {
	results := make([]ToolResult, len(actions))
	semaphore := make(chan struct{}, o.maxConcurrency)
	var wg sync.WaitGroup

	for i, action := range actions {
		wg.Add(1)
		go func(index int, action llm.ToolCall) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()
			results[index] = o.executeSingle(ctx, action)
		}(i, action)
	}
	wg.Wait()
	return results
}

// executeSingle executes one MCP tool call with a timeout.
func (o *Orchestrator) executeSingle(ctx context.Context, action llm.ToolCall) ToolResult {
	callCtx, cancel := context.WithTimeout(ctx, o.toolTimeout)
	defer cancel()

	content, err := o.client.CallTool(callCtx, action.Name, action.Arguments)
	if err != nil {
		o.log.Error("MCP tool call failed", zap.String("tool", action.Name), zap.Error(err))
		return ToolResult{ToolKey: action.Name, Success: false, Content: "Error: " + err.Error()}
	}

	o.log.Info("MCP tool returned", zap.String("tool", action.Name), zap.String("content", content))

	var parsed map[string]any
	if json.Unmarshal([]byte(content), &parsed) == nil {
		if _, hasError := parsed["error"]; hasError {
			return ToolResult{ToolKey: action.Name, Success: false, Content: content}
		}
	}

	return ToolResult{ToolKey: action.Name, Success: true, Content: content}
}

// buildResultPrompt builds the follow-up prompt with the latest tool results.
func (o *Orchestrator) buildResultPrompt(originalPrompt string, results []ToolResult) string {
	var lines []string
	for _, result := range results {
		status := "OK"
		if !result.Success {
			status = "FAILED"
		}
		lines = append(lines, fmt.Sprintf("[%s] %s: %s", result.ToolKey, status, result.Content))
	}

	return fmt.Sprintf(
		"%s\n\n[Tool Results]\n%s\n\n[Next Step]\n"+
			"Do NOT re-call tools you have already called successfully. "+
			"If all needed info is available, respond with your final actions. "+
			"Otherwise call only the necessary tools in one batch.\n",
		originalPrompt, strings.Join(lines, "\n"))
}

// callSignature returns a unique string for an action based on its name and arguments.
func (o *Orchestrator) callSignature(action llm.ToolCall) string {
	args, _ := json.Marshal(action.Arguments)
	return action.Name + "|" + string(args)
}

// isMCPSchema reports whether an OpenAI function schema is an MCP tool schema.
func isMCPSchema(schema map[string]any) bool {
	fn, ok := schema["function"].(map[string]any)
	if !ok {
		return false
	}
	name, _ := fn["name"].(string)
	return strings.HasPrefix(name, "mcp_")
}
