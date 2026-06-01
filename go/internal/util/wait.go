package util

import (
	"context"
	"time"
)

// Sleep waits for d or until ctx is cancelled, whichever comes first. It returns
// false if ctx was cancelled before d elapsed, and true otherwise.
func Sleep(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
