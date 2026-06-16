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

func resetTTSState(t *testing.T) {
	t.Helper()
	Suppressed.Store(false)
	Interrupt.Store(false)
	Speaking.Store(false)
	pending.Store(0)
	t.Cleanup(func() {
		Suppressed.Store(false)
		Interrupt.Store(false)
		Speaking.Store(false)
		pending.Store(0)
	})
}

func TestSetSuppressedMuteWhileIdle(t *testing.T) {
	resetTTSState(t)

	SetSuppressed(true)
	require.True(t, Suppressed.Load(), "TTS is muted")
	require.False(t, Interrupt.Load(),
		"muting while idle must not leave a lingering interrupt that would drop the next utterance")
}

func TestSetSuppressedMuteWhileSpeaking(t *testing.T) {
	resetTTSState(t)
	Speaking.Store(true)

	SetSuppressed(true)
	require.True(t, Interrupt.Load(), "muting during active playback interrupts it")
}

func TestSetSuppressedMuteWhileQueued(t *testing.T) {
	resetTTSState(t)
	pending.Store(1)

	SetSuppressed(true)
	require.True(t, Interrupt.Load(), "muting with queued speech interrupts/drops it")
}

func TestSetSuppressedUnmuteClearsStaleInterrupt(t *testing.T) {
	resetTTSState(t)
	Suppressed.Store(true)
	Interrupt.Store(true)

	SetSuppressed(false)
	require.False(t, Suppressed.Load(), "TTS is unmuted")
	require.False(t, Interrupt.Load(), "unmuting clears the stale interrupt so the next utterance plays")
}
