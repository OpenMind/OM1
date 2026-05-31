package providers

import (
	"sync"

	"github.com/gordonklaus/portaudio"
)

// PortAudioRef reference-counts the process-global PortAudio library so that
// overlapping audio consumers (e.g. the outgoing and incoming modes' sensors
// during a mode transition) share a single Initialize/Terminate pair.
//
// This state is intentionally a package-level singleton rather than something
// held per consumer: portaudio.Initialize/Terminate act on one process-wide
// library instance, so a per-consumer counter would let one consumer terminate
// the library out from under another — freeing it mid-read and segfaulting
// inside cgo.
type PortAudioRef struct {
	mu       sync.Mutex
	refCount int
}

// PortAudio is the single shared reference counter for the PortAudio library.
var PortAudio PortAudioRef

// Acquire initializes PortAudio on the first active consumer and increments the
// reference count for subsequent ones.
func (p *PortAudioRef) Acquire() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.refCount == 0 {
		if err := portaudio.Initialize(); err != nil {
			return err
		}
	}
	p.refCount++
	return nil
}

// Release decrements the reference count and terminates PortAudio once the last
// consumer has released it.
func (p *PortAudioRef) Release() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.refCount == 0 {
		return
	}
	p.refCount--
	if p.refCount == 0 {
		portaudio.Terminate()
	}
}
