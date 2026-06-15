package hooks

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLumaHookRegistered(t *testing.T) {
	_, ok := lookupHook("luma_hook", "luma_intro_hook")
	require.True(t, ok)
}

func TestLumaPromptMentionsLuma(t *testing.T) {
	require.True(t, strings.Contains(defaultLumaPrompt, "{help_message}"),
		"prompt must interpolate the help message")
	require.Contains(t, strings.ToLower(defaultLumaHelp), "luma")
}
