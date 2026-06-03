package actions

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

type recordingConnector struct {
	mu      *sync.Mutex
	order   *[]string
	label   string
	output  Output
	err     error
	delay   time.Duration
	tickCnt atomic.Int32
	stopCnt atomic.Int32
}

func (c *recordingConnector) Connect(_ context.Context, _ Input) (Output, error) {
	if c.delay > 0 {
		time.Sleep(c.delay)
	}
	c.mu.Lock()
	*c.order = append(*c.order, c.label)
	c.mu.Unlock()
	return c.output, c.err
}

func (c *recordingConnector) Tick(context.Context) { c.tickCnt.Add(1) }
func (c *recordingConnector) Stop()                { c.stopCnt.Add(1) }

func newAction(label string, conn Connector) *AgentAction {
	return &AgentAction{Name: label, LLMLabel: label, Connector: conn}
}

func TestOrchestratorDefaultsToConcurrent(t *testing.T) {
	o := NewOrchestrator(nil, "", nil, zap.NewNop())
	require.Equal(t, Concurrent, o.mode)
}

func TestSubmitConcurrentReturnsAllResults(t *testing.T) {
	var mu sync.Mutex
	var order []string
	a := newAction("a", &recordingConnector{mu: &mu, order: &order, label: "a", output: "ra"})
	b := newAction("b", &recordingConnector{mu: &mu, order: &order, label: "b", err: errors.New("fail")})

	o := NewOrchestrator([]*AgentAction{a, b}, Concurrent, nil, zap.NewNop())
	results := o.Submit(context.Background(), []Call{{Action: a}, {Action: b}})

	require.Len(t, results, 2)
	require.Equal(t, "a", results[0].ActionName)
	require.Equal(t, "ra", results[0].Output)
	require.NoError(t, results[0].Err)
	require.Error(t, results[1].Err)
}

func TestSubmitSequentialPreservesOrder(t *testing.T) {
	var mu sync.Mutex
	var order []string
	a := newAction("a", &recordingConnector{mu: &mu, order: &order, label: "a", delay: 5 * time.Millisecond})
	b := newAction("b", &recordingConnector{mu: &mu, order: &order, label: "b"})

	o := NewOrchestrator([]*AgentAction{a, b}, Sequential, nil, zap.NewNop())
	results := o.Submit(context.Background(), []Call{{Action: a}, {Action: b}})

	require.Len(t, results, 2)
	require.Equal(t, []string{"a", "b"}, order, "sequential runs in call order")
}

func TestSubmitWithDepsRespectsOrder(t *testing.T) {
	var mu sync.Mutex
	var order []string
	parent := newAction("parent", &recordingConnector{mu: &mu, order: &order, label: "parent", delay: 10 * time.Millisecond})
	child := newAction("child", &recordingConnector{mu: &mu, order: &order, label: "child"})

	deps := map[string][]string{"child": {"parent"}}
	o := NewOrchestrator([]*AgentAction{parent, child}, WithDeps, deps, zap.NewNop())
	results := o.Submit(context.Background(), []Call{{Action: child}, {Action: parent}})

	require.Len(t, results, 2)
	require.Equal(t, []string{"parent", "child"}, order, "dependency runs before dependent")
}

func TestParseCalls(t *testing.T) {
	a := newAction("speak", stubConnector{})
	o := NewOrchestrator([]*AgentAction{a}, Concurrent, nil, zap.NewNop())

	calls, err := o.ParseCalls([]map[string]any{
		{"name": "speak", "arguments": map[string]any{"text": "hi"}},
	})
	require.NoError(t, err)
	require.Len(t, calls, 1)
	require.Same(t, a, calls[0].Action)
	require.Equal(t, map[string]any{"text": "hi"}, calls[0].Input)
}

func TestParseCallsUnknownAction(t *testing.T) {
	o := NewOrchestrator(nil, Concurrent, nil, zap.NewNop())
	_, err := o.ParseCalls([]map[string]any{{"name": "ghost"}})
	require.Error(t, err)
	require.Contains(t, err.Error(), "ghost")
}

func TestStartTicksAndStopsOnCancel(t *testing.T) {
	conn := &recordingConnector{mu: &sync.Mutex{}, order: &[]string{}, label: "a"}
	a := newAction("a", conn)
	o := NewOrchestrator([]*AgentAction{a}, Concurrent, nil, zap.NewNop())

	ctx, cancel := context.WithCancel(context.Background())
	done := o.Start(ctx)
	time.Sleep(20 * time.Millisecond)
	cancel()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Start did not return after cancellation")
	}
	require.Positive(t, conn.tickCnt.Load(), "Tick was called at least once")
	require.Equal(t, int32(1), conn.stopCnt.Load(), "Stop is called exactly once on shutdown")
}
