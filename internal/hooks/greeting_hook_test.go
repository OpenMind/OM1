package hooks

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEndMessageDefaults(t *testing.T) {
	cases := []struct {
		name          string
		turnCount     int
		maxTurnCount  int
		wantSubstring string
	}{
		{"max turns reached", 3, 3, defaultEndMessageMaxTurns},
		{"beyond max turns", 5, 3, defaultEndMessageMaxTurns},
		{"active conversation", 1, 3, defaultEndMessageActive},
		{"no turns", 0, 3, defaultEndMessageNoTurns},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := endMessage(map[string]any{}, nil, tc.turnCount, tc.maxTurnCount)
			require.Equal(t, tc.wantSubstring, got)
		})
	}
}

func TestEndMessageConfigOverrides(t *testing.T) {
	cfg := map[string]any{
		"end_message_max_turns": "bye after max",
		"end_message_active":    "bye after chat",
		"end_message_no_turns":  "bye no chat",
	}

	require.Equal(t, "bye after max", endMessage(cfg, nil, 3, 3))
	require.Equal(t, "bye after chat", endMessage(cfg, nil, 1, 3))
	require.Equal(t, "bye no chat", endMessage(cfg, nil, 0, 3))
}

func TestEndMessageBlankOverrideFallsBackToDefault(t *testing.T) {
	// A present-but-empty config value should fall back to the default rather
	// than announcing an empty farewell.
	cfg := map[string]any{"end_message_active": "   "}
	require.Equal(t, defaultEndMessageActive, endMessage(cfg, nil, 1, 3))
}

func TestEndMessageAppliesTemplateVars(t *testing.T) {
	cfg := map[string]any{"end_message_no_turns": "Goodbye from {robot_name}!"}
	got := endMessage(cfg, map[string]any{"robot_name": "Bits"}, 0, 3)
	require.Equal(t, "Goodbye from Bits!", got)
}
