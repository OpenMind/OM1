package tts

import "sync/atomic"

var Speaking atomic.Bool

var Interrupt atomic.Bool

var generation atomic.Uint64

var pending atomic.Int64

func Busy() bool {
	return Speaking.Load() || pending.Load() > 0
}

func RequestInterrupt() {
	generation.Add(1)
	Interrupt.Store(true)
}
