package qr_scanner

import (
	"sync"
	"time"
)

// debouncer drops repeated `pk` values seen within a sliding time window.
type debouncer struct {
	mu     sync.Mutex
	window time.Duration
	seen   map[string]time.Time
	now    func() time.Time
}

func newDebouncer(window time.Duration) *debouncer {
	return &debouncer{
		window: window,
		seen:   make(map[string]time.Time),
		now:    time.Now,
	}
}

// TryRecord returns true if pk was not seen within the configured window.
// Recording also prunes entries older than 10x the window to bound memory.
func (d *debouncer) TryRecord(pk string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	now := d.now()
	d.pruneLocked(now)

	if last, ok := d.seen[pk]; ok && now.Sub(last) < d.window {
		return false
	}
	d.seen[pk] = now
	return true
}

func (d *debouncer) pruneLocked(now time.Time) {
	cutoff := now.Add(-10 * d.window)
	for k, t := range d.seen {
		if t.Before(cutoff) {
			delete(d.seen, k)
		}
	}
}
