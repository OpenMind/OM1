package tts

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSpeakingFlag(t *testing.T) {
	t.Cleanup(func() { Speaking.Store(false) })

	require.False(t, Speaking.Load(), "defaults to not speaking")
	Speaking.Store(true)
	require.True(t, Speaking.Load())
	Speaking.Store(false)
	require.False(t, Speaking.Load())
}
