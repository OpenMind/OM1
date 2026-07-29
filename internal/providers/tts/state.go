package tts

import "sync/atomic"

var Speaking atomic.Bool

var Interrupt atomic.Bool

var generation atomic.Uint64

var pending atomic.Int64

var Suppressed atomic.Bool

func Busy() bool {
	return Speaking.Load() || pending.Load() > 0
}

func RequestInterrupt() {
	generation.Add(1)
	Interrupt.Store(true)
}

func SetSuppressed(suppressed bool) {
	Suppressed.Store(suppressed)
	switch {
	case suppressed && Busy():
		RequestInterrupt()
	case !suppressed:
		Interrupt.Store(false)
	}
}

// BargeIn lifts any mute and cuts off whatever is currently being said, so the
// caller's next utterance starts immediately instead of queueing behind it.
//
// The ordering is deliberate. Unmuting clears the interrupt flag (see
// SetSuppressed), so the mute must be lifted before the interrupt is raised, or
// the barge-in is silently swallowed. Equally, the interrupt is only armed while
// the player is busy: a sticky flag with no utterance to consume it would be
// eaten by whatever spoke next.
func BargeIn() {
	Suppressed.Store(false)
	Interrupt.Store(false)

	if Busy() {
		RequestInterrupt()
	}
}
