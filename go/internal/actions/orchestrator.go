package actions

import (
	"context"
	"fmt"
	"sync"

	"go.uber.org/zap"
)

type ExecMode string

const (
	Concurrent ExecMode = "concurrent"
	Sequential ExecMode = "sequential"
	WithDeps   ExecMode = "dependencies"
)

type Call struct {
	Action *AgentAction
	Input  Input
}

type Result struct {
	ActionName string
	Output     Output
	Err        error
}

type Orchestrator struct {
	actions map[string]*AgentAction
	mode    ExecMode
	deps    map[string][]string
	log     *zap.Logger
	wg      sync.WaitGroup
}

func NewOrchestrator(
	agentActions []*AgentAction,
	mode ExecMode,
	dependencies map[string][]string,
	log *zap.Logger,
) *Orchestrator {
	actionByLabel := make(map[string]*AgentAction, len(agentActions))
	for _, action := range agentActions {
		actionByLabel[action.LLMLabel] = action
	}
	if mode == "" {
		mode = Concurrent
	}
	return &Orchestrator{actions: actionByLabel, mode: mode, deps: dependencies, log: log}
}

// Submit executes the given Calls and returns their Results.
func (o *Orchestrator) Submit(ctx context.Context, calls []Call) []Result {
	switch o.mode {
	case Sequential:
		return o.runSequential(ctx, calls)
	case WithDeps:
		return o.runWithDeps(ctx, calls)
	default:
		return o.runConcurrent(ctx, calls)
	}
}

// Start runs the Tick loop for all connectors and waits for them to finish when the context is canceled.
func (o *Orchestrator) Start(ctx context.Context) <-chan struct{} {
	done := make(chan struct{})

	for _, agentAction := range o.actions {
		o.wg.Add(1)
		go func(connector Connector) {
			defer o.wg.Done()
			defer connector.Stop()

			for {
				select {
				case <-ctx.Done():
					return
				default:
					connector.Tick(ctx)
				}
			}
		}(agentAction.Connector)
	}

	go func() {
		o.wg.Wait()
		close(done)
	}()

	return done
}

func (o *Orchestrator) runConcurrent(ctx context.Context, calls []Call) []Result {
	results := make([]Result, len(calls))
	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Add(1)
		go func(callIndex int, call Call) {
			defer wg.Done()
			output, err := call.Action.Connector.Connect(ctx, call.Input)
			results[callIndex] = Result{ActionName: call.Action.LLMLabel, Output: output, Err: err}
		}(i, call)
	}
	wg.Wait()
	return results
}

func (o *Orchestrator) runSequential(ctx context.Context, calls []Call) []Result {
	results := make([]Result, 0, len(calls))
	for _, call := range calls {
		output, err := call.Action.Connector.Connect(ctx, call.Input)
		results = append(results, Result{ActionName: call.Action.LLMLabel, Output: output, Err: err})
		if ctx.Err() != nil {
			break
		}
	}
	return results
}

func (o *Orchestrator) runWithDeps(ctx context.Context, calls []Call) []Result {
	indexByLabel := make(map[string]int, len(calls))
	for i, call := range calls {
		indexByLabel[call.Action.LLMLabel] = i
	}

	results := make([]Result, len(calls))
	completed := make([]bool, len(calls))
	var mu sync.Mutex

	var runCall = func(callIndex int) {
		call := calls[callIndex]
		for _, dependency := range o.deps[call.Action.LLMLabel] {
			dependencyIndex, ok := indexByLabel[dependency]
			if !ok {
				continue
			}
			for {
				mu.Lock()
				ready := completed[dependencyIndex]
				mu.Unlock()
				if ready {
					break
				}
				if ctx.Err() != nil {
					return
				}
			}
		}
		output, err := call.Action.Connector.Connect(ctx, call.Input)
		mu.Lock()
		results[callIndex] = Result{ActionName: call.Action.LLMLabel, Output: output, Err: err}
		completed[callIndex] = true
		mu.Unlock()
	}

	var wg sync.WaitGroup
	for i := range calls {
		wg.Add(1)
		go func(callIndex int) {
			defer wg.Done()
			runCall(callIndex)
		}(i)
	}
	wg.Wait()
	return results
}

// ParseCalls converts raw LLM tool calls to Calls with AgentActions looked up.
func (o *Orchestrator) ParseCalls(rawToolCalls []map[string]any) ([]Call, error) {
	var calls []Call
	for _, rawToolCall := range rawToolCalls {
		actionName, _ := rawToolCall["name"].(string)
		arguments, _ := rawToolCall["arguments"].(map[string]any)

		agentAction, ok := o.actions[actionName]
		if !ok {
			return nil, fmt.Errorf("unknown action %q", actionName)
		}
		calls = append(calls, Call{Action: agentAction, Input: arguments})
	}
	return calls, nil
}
