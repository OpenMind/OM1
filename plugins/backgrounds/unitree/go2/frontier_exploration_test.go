package go2

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
	zenohsession "github.com/openmind/om1/internal/zenoh"
)

func newTestFrontierExploration(contextAwareText map[string]any) *UnitreeGo2FrontierExploration {
	return &UnitreeGo2FrontierExploration{
		log:              zap.NewNop(),
		contextAwareText: contextAwareText,
	}
}

func stdMsgsStringPayload(s string) []byte {
	buf := []byte{0x00, 0x01, 0x00, 0x00}
	return zenohsession.AppendCDRString(buf, s)
}

func TestFrontierExplorationConfigDefaults(t *testing.T) {
	b, err := NewUnitreeGo2FrontierExploration(nil)
	require.NoError(t, err)
	e := b.(*UnitreeGo2FrontierExploration)
	require.Equal(t, map[string]any{"exploration_done": true}, e.contextAwareText)
	e.Stop()
}

func TestFrontierExplorationConfigCustomContext(t *testing.T) {
	b, err := NewUnitreeGo2FrontierExploration(map[string]any{
		"topic":              "nav/explore",
		"context_aware_text": `{"done": false, "progress": 50}`,
	})
	require.NoError(t, err)
	e := b.(*UnitreeGo2FrontierExploration)
	require.Equal(t, false, e.contextAwareText["done"])
	require.EqualValues(t, 50, e.contextAwareText["progress"])
	e.Stop()
}

func TestFrontierExplorationConfigInvalidJSONFallsBack(t *testing.T) {
	b, err := NewUnitreeGo2FrontierExploration(map[string]any{
		"context_aware_text": "not valid json{{{",
	})
	require.NoError(t, err)
	e := b.(*UnitreeGo2FrontierExploration)
	require.Equal(t, map[string]any{"exploration_done": true}, e.contextAwareText)
	e.Stop()
}

func TestFrontierExplorationCompletePublishesContext(t *testing.T) {
	ctxText := map[string]any{"exploration_done": true}
	e := newTestFrontierExploration(ctxText)

	drainModeContext()

	e.onStatus(stdMsgsStringPayload(`{"complete": true, "info": "all done"}`))

	select {
	case got := <-providers.ModeContext().Updates():
		require.Equal(t, ctxText, got)
	case <-time.After(time.Second):
		t.Fatal("expected a context update when exploration completes")
	}
}

func TestFrontierExplorationIncompleteDoesNotPublish(t *testing.T) {
	e := newTestFrontierExploration(map[string]any{"exploration_done": true})
	drainModeContext()

	e.onStatus(stdMsgsStringPayload(`{"complete": false, "info": "still going"}`))

	select {
	case got := <-providers.ModeContext().Updates():
		t.Fatalf("did not expect a context update for incomplete exploration, got %v", got)
	case <-time.After(100 * time.Millisecond):
	}
}

func TestFrontierExplorationIgnoresBadPayloads(t *testing.T) {
	e := newTestFrontierExploration(map[string]any{"exploration_done": true})
	drainModeContext()

	e.onStatus(nil)
	e.onStatus([]byte{0x00, 0x01})
	e.onStatus(stdMsgsStringPayload("not json"))

	select {
	case got := <-providers.ModeContext().Updates():
		t.Fatalf("did not expect a context update for malformed payloads, got %v", got)
	case <-time.After(100 * time.Millisecond):
	}
}

func drainModeContext() {
	for {
		select {
		case <-providers.ModeContext().Updates():
		default:
			return
		}
	}
}
