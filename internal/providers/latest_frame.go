package providers

import (
	"sync"
	"time"
)

type LatestFrameProvider struct {
	mu   sync.RWMutex
	jpeg []byte
	ts   time.Time
}

var (
	latestFrameOnce     sync.Once
	latestFrameInstance *LatestFrameProvider
)

// LatestFrame returns the singleton LatestFrameProvider.
func LatestFrame() *LatestFrameProvider {
	latestFrameOnce.Do(func() { latestFrameInstance = &LatestFrameProvider{} })
	return latestFrameInstance
}

// Set stores a copy of the most recent JPEG frame and its capture time. Empty
// frames are ignored.
func (p *LatestFrameProvider) Set(jpeg []byte, ts time.Time) {
	if len(jpeg) == 0 {
		return
	}
	buf := make([]byte, len(jpeg))
	copy(buf, jpeg)
	if ts.IsZero() {
		ts = time.Now()
	}
	p.mu.Lock()
	p.jpeg = buf
	p.ts = ts
	p.mu.Unlock()
}

// Get returns a copy of the most recent JPEG frame and its capture time. ok is
// false when no frame has been captured yet.
func (p *LatestFrameProvider) Get() (jpeg []byte, ts time.Time, ok bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(p.jpeg) == 0 {
		return nil, time.Time{}, false
	}
	buf := make([]byte, len(p.jpeg))
	copy(buf, p.jpeg)
	return buf, p.ts, true
}

// GetFresh behaves like Get but returns ok=false when the latest frame is older
// than maxAge. A non-positive maxAge disables the staleness check.
func (p *LatestFrameProvider) GetFresh(maxAge time.Duration) (jpeg []byte, ts time.Time, ok bool) {
	jpeg, ts, ok = p.Get()
	if !ok {
		return nil, time.Time{}, false
	}
	if maxAge > 0 && time.Since(ts) > maxAge {
		return nil, time.Time{}, false
	}
	return jpeg, ts, true
}
