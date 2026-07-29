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


func TestBargeInInterruptsWhileSpeaking(t *testing.T) {
	resetTTSState(t)
	Speaking.Store(true)
	before := generation.Load()

	BargeIn()
	require.True(t, Interrupt.Load(), "the in-flight utterance is interrupted")
	require.False(t, Suppressed.Load(), "the mute is lifted so the new utterance can play")
	require.Greater(t, generation.Load(), before, "the generation bump drops anything still queued")
}

func TestBargeInWhileMutedStillInterrupts(t *testing.T) {
	resetTTSState(t)
	Suppressed.Store(true)
	Speaking.Store(true)

	// The ordering trap: unmuting clears Interrupt, so lifting the mute after
	// raising the interrupt would swallow the barge-in and leave the caller's
	// new utterance queued behind the one it was meant to replace.
	BargeIn()
	require.False(t, Suppressed.Load())
	require.True(t, Interrupt.Load(),
		"the interrupt must survive the unmute, or the barge-in never happens")
}

func TestBargeInWhileIdleDoesNotArmInterrupt(t *testing.T) {
	resetTTSState(t)
	Suppressed.Store(true)
	Interrupt.Store(true)

	BargeIn()
	require.False(t, Suppressed.Load(), "the mute is still lifted")
	require.False(t, Interrupt.Load(),
		"no stale interrupt is left to swallow the utterance about to be queued")
}

func TestBargeInWhileQueued(t *testing.T) {
	resetTTSState(t)
	pending.Store(1)

	BargeIn()
	require.True(t, Interrupt.Load(), "queued-but-unplayed speech still counts as busy")
}
