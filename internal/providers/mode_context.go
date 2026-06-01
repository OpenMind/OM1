package providers

import "sync"

// ModeContextProvider is an in-process bus carrying user-context updates that
// drive context-aware mode transitions. Publishers (backgrounds, actions) push
// updates via Publish; the cortex loop selects on Updates, so a transition is
// re-evaluated the instant an update arrives rather than on the next (possibly
// far-off) tick.
type ModeContextProvider struct {
	ch chan map[string]any
}

var (
	modeContextOnce     sync.Once
	modeContextInstance *ModeContextProvider
)

// modeContextBuffer is the number of pending updates the bus holds before it
// starts dropping. Context updates are level-triggered flags (e.g. set the flag
// to true), so dropping a duplicate is harmless.
const modeContextBuffer = 64

// ModeContext returns the singleton mode-context bus.
func ModeContext() *ModeContextProvider {
	modeContextOnce.Do(func() {
		modeContextInstance = &ModeContextProvider{ch: make(chan map[string]any, modeContextBuffer)}
	})
	return modeContextInstance
}

// Publish queues a context update for the cortex loop. It is best-effort and
// non-blocking: if the buffer is full the update is dropped rather than blocking
// the caller (publishers run on latency-sensitive paths).
func (p *ModeContextProvider) Publish(update map[string]any) {
	if len(update) == 0 {
		return
	}
	select {
	case p.ch <- update:
	default:
	}
}

// Updates returns the receive end of the bus, consumed by the cortex loop.
func (p *ModeContextProvider) Updates() <-chan map[string]any {
	return p.ch
}
