package greeting_conversation

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/openmind/om1/internal/providers"
)

func buildConnector(t *testing.T, cfg map[string]any) *Connector {
	t.Helper()
	if cfg == nil {
		cfg = map[string]any{}
	}
	cfg["api_key"] = "test-key"

	conn, err := NewElevenLabsGreetingConversation(cfg)
	require.NoError(t, err)
	t.Cleanup(conn.Stop)

	c, ok := conn.(*Connector)
	require.True(t, ok, "expected *Connector")
	return c
}

func TestTickIntervalConfigurable(t *testing.T) {
	cases := []struct {
		name string
		cfg  map[string]any
		want time.Duration
	}{
		{"unset uses default", nil, defaultTickInterval},
		{"custom fractional", map[string]any{"tick_interval_sec": 0.05}, 50 * time.Millisecond},
		{"custom whole", map[string]any{"tick_interval_sec": 3}, 3 * time.Second},
		{"zero uses default", map[string]any{"tick_interval_sec": 0}, defaultTickInterval},
		{"negative uses default", map[string]any{"tick_interval_sec": -1}, defaultTickInterval},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := buildConnector(t, tc.cfg)
			require.Equal(t, tc.want, c.tickInterval)
		})
	}
}

func TestTickHonorsConfiguredInterval(t *testing.T) {
	c := buildConnector(t, map[string]any{"tick_interval_sec": 0.05})

	start := time.Now()
	c.Tick(context.Background())
	require.GreaterOrEqual(t, time.Since(start), 30*time.Millisecond,
		"Tick should wait roughly the configured 50ms interval before advancing")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start = time.Now()
	c.Tick(ctx)
	require.Less(t, time.Since(start), 30*time.Millisecond,
		"Tick should return immediately when its context is already cancelled")
}

func TestFinishedTriggersModeSwitch(t *testing.T) {
	drainModeContext()

	c := buildConnector(t, map[string]any{"tick_interval_sec": 0.05})

	_, err := c.Connect(context.Background(), map[string]any{
		"conversation_state": string(providers.StateFinished),
		"response":           "",
	})
	require.NoError(t, err)

	select {
	case update := <-providers.ModeContext().Updates():
		require.Equal(t, true, update["greeting_conversation_finished"],
			"finished turn should publish greeting_conversation_finished=true")
	case <-time.After(2 * time.Second):
		t.Fatal("expected greeting_conversation_finished context update after a finished turn")
	}
}

func drainModeContext() {
	ch := providers.ModeContext().Updates()
	for {
		select {
		case <-ch:
		default:
			return
		}
	}
}
