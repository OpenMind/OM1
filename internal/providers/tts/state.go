package tts

import "sync/atomic"

var Speaking atomic.Bool

var Priority atomic.Bool

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
