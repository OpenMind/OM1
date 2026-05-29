package actions

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

// testConnector records how many times Tick and Stop are called.
type testConnector struct {
	tickCount atomic.Int64
	stopCount atomic.Int64
}

func (c *testConnector) Connect(_ context.Context, _ Input) (Output, error) { return nil, nil }

func (c *testConnector) Tick(ctx context.Context) {
	c.tickCount.Add(1)
	// Block until ctx is cancelled so we don't spin.
	select {
	case <-ctx.Done():
	case <-time.After(10 * time.Second):
	}
}

func (c *testConnector) Stop() { c.stopCount.Add(1) }

func TestStartTickers_LaunchesGoroutines(t *testing.T) {
	connectorA := &testConnector{}
	connectorB := &testConnector{}

	orchestrator := NewOrchestrator(
		[]*AgentAction{
			{LLMLabel: "a", Connector: connectorA},
			{LLMLabel: "b", Connector: connectorB},
		},
		Concurrent, nil, nil,
	)

	ctx, cancel := context.WithCancel(context.Background())

	done := orchestrator.Start(ctx)

	// Both goroutines should enter Tick almost immediately.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if connectorA.tickCount.Load() >= 1 && connectorB.tickCount.Load() >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if connectorA.tickCount.Load() < 1 {
		t.Error("connectorA.Tick was never called")
	}
	if connectorB.tickCount.Load() < 1 {
		t.Error("connectorB.Tick was never called")
	}

	// Cancel the context — goroutines should call Stop and exit.
	cancel()

	deadline = time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if connectorA.stopCount.Load() >= 1 && connectorB.stopCount.Load() >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if connectorA.stopCount.Load() != 1 {
		t.Errorf("connectorA.Stop called %d times, want 1", connectorA.stopCount.Load())
	}
	if connectorB.stopCount.Load() != 1 {
		t.Errorf("connectorB.Stop called %d times, want 1", connectorB.stopCount.Load())
	}

	// Verify the done channel closes after all goroutines exit
	select {
	case <-done:
		// Success
	case <-time.After(500 * time.Millisecond):
		t.Error("done channel was not closed after context cancellation")
	}
}
