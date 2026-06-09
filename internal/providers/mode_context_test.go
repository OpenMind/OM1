package providers

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func newModeContext(buf int) *ModeContextProvider {
	return &ModeContextProvider{ch: make(chan map[string]any, buf)}
}

func TestModeContextSingleton(t *testing.T) {
	require.Same(t, ModeContext(), ModeContext())
}

func TestModeContextPublishDelivers(t *testing.T) {
	p := newModeContext(4)
	p.Publish(map[string]any{"person_near": true})

	select {
	case got := <-p.Updates():
		require.Equal(t, true, got["person_near"])
	default:
		t.Fatal("expected a published update to be received")
	}
}

func TestModeContextPublishIgnoresEmpty(t *testing.T) {
	p := newModeContext(4)
	p.Publish(nil)
	p.Publish(map[string]any{})
	select {
	case <-p.Updates():
		t.Fatal("empty updates must not be queued")
	default:
	}
}

func TestModeContextDropsWhenFull(t *testing.T) {
	p := newModeContext(1)
	p.Publish(map[string]any{"a": 1})
	require.NotPanics(t, func() { p.Publish(map[string]any{"b": 2}) }, "full buffer drops rather than blocks")
	require.Len(t, p.Updates(), 1, "only the first update is retained")
}
