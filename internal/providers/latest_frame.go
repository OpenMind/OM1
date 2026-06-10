package providers

import (
	"context"
	"sync"
	"time"
)

type LatestFrameProvider struct {
	mu     sync.RWMutex
	jpeg   []byte
	ts     time.Time
	notify chan struct{}
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

	if p.notify != nil {
		close(p.notify)
		p.notify = nil
	}

	p.mu.Unlock()
}

// freshLocked is a helper for WaitForFresh that checks if the current frame satisfies maxAge while holding the lock.
func (p *LatestFrameProvider) freshLocked(maxAge time.Duration) (jpeg []byte, ts time.Time, ok bool, wait <-chan struct{}) {
	if len(p.jpeg) > 0 && (maxAge <= 0 || time.Since(p.ts) <= maxAge) {
		buf := make([]byte, len(p.jpeg))
		copy(buf, p.jpeg)
		return buf, p.ts, true, nil
	}
	if p.notify == nil {
		p.notify = make(chan struct{})
	}
	return nil, time.Time{}, false, p.notify
}

// Get returns a copy of the most recent JPEG frame and its capture time.
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

// GetFresh behaves like Get but returns ok=false when the latest frame is older than maxAge.
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

// WaitForFresh waits for and returns the next frame that satisfies maxAge, or returns false if the context is canceled or the timeout is reached first.
func (p *LatestFrameProvider) WaitForFresh(ctx context.Context, maxAge, timeout time.Duration) (jpeg []byte, ts time.Time, ok bool) {
	deadline := time.After(timeout)
	for {
		p.mu.Lock()
		jpeg, ts, ok, wait := p.freshLocked(maxAge)
		p.mu.Unlock()
		if ok {
			return jpeg, ts, true
		}
		if timeout <= 0 {
			return nil, time.Time{}, false
		}
		select {
		case <-wait:
		case <-deadline:
			return nil, time.Time{}, false
		case <-ctx.Done():
			return nil, time.Time{}, false
		}
	}
}
