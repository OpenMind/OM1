package tts

import "sync/atomic"

var Speaking atomic.Bool

var Interrupt atomic.Bool

var generation atomic.Uint64

func RequestInterrupt() {
	generation.Add(1)
	Interrupt.Store(true)
}
