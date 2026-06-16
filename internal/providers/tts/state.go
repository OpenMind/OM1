package tts

import "sync/atomic"

var Speaking atomic.Bool

var Interrupt atomic.Bool

var generation atomic.Uint64

var pending atomic.Int64

// Suppressed, when true, mutes all TTS output.
var Suppressed atomic.Bool

func Busy() bool {
	return Speaking.Load() || pending.Load() > 0
}

func RequestInterrupt() {
	generation.Add(1)
	Interrupt.Store(true)
}

// SetSuppressed mutes or unmutes TTS. Muting also interrupts in-flight speech.
func SetSuppressed(suppressed bool) {
	Suppressed.Store(suppressed)
	if suppressed {
		RequestInterrupt()
	}
}
