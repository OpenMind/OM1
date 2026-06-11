package vlm

import (
	"sync"
	"time"
)

// LatestDescriptionProvider is a singleton that holds the most recent VLM description and its timestamp.
type LatestDescriptionProvider struct {
	mu   sync.RWMutex
	text string
	ts   time.Time
}

var (
	latestDescriptionOnce     sync.Once
	latestDescriptionInstance *LatestDescriptionProvider
)

// LatestDescription returns the singleton LatestDescriptionProvider.
func LatestDescription() *LatestDescriptionProvider {
	latestDescriptionOnce.Do(func() { latestDescriptionInstance = &LatestDescriptionProvider{} })
	return latestDescriptionInstance
}

// Set stores the most recent description and the time it was produced. Empty
// descriptions are ignored.
func (p *LatestDescriptionProvider) Set(text string, ts time.Time) {
	if text == "" {
		return
	}
	if ts.IsZero() {
		ts = time.Now()
	}
	p.mu.Lock()
	p.text = text
	p.ts = ts
	p.mu.Unlock()
}

// Get returns the most recent description and the time it was produced. ok is
// false when no description has been produced yet.
func (p *LatestDescriptionProvider) Get() (text string, ts time.Time, ok bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.text == "" {
		return "", time.Time{}, false
	}
	return p.text, p.ts, true
}

// GetFresh behaves like Get but returns ok=false when the latest description is
// older than maxAge. A non-positive maxAge disables the staleness check.
func (p *LatestDescriptionProvider) GetFresh(maxAge time.Duration) (text string, ts time.Time, ok bool) {
	text, ts, ok = p.Get()
	if !ok {
		return "", time.Time{}, false
	}
	if maxAge > 0 && time.Since(ts) > maxAge {
		return "", time.Time{}, false
	}
	return text, ts, true
}
