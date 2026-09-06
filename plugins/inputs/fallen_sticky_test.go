package inputs

import (
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers"
)

// newTestTracker builds a tracker with deterministic latch settings for classify tests.
func newTestTracker(clearStreak int, minHold, arrivedRelease time.Duration) *fallenPersonTracker {
	return &fallenPersonTracker{
		log:            zap.NewNop(),
		clearStreak:    clearStreak,
		minHold:        minHold,
		lockWidth:      0.5,
		arrivedRelease: arrivedRelease,
	}
}

func TestStickyLockHoldsThroughDropout(t *testing.T) {
	// Reset shared state and isolate from other tests.
	providers.SetPersonDownAlert(false)
	providers.SetPersonDownArrived(false)
	defer func() {
		providers.SetPersonDownAlert(false)
		providers.SetPersonDownArrived(false)
	}()

	s := newTestTracker(2, 0, 10*time.Second)
	present := providers.FallenSnapshot{Present: true, Alert: true, Name: "wendy", WidthFrac: 0.6}
	absent := providers.FallenSnapshot{Present: false}

	// Alert asserts.
	s.classify(present)
	if !providers.PersonDownAlert() {
		t.Fatal("alert should be set after a present+alert frame")
	}

	// Simulate the connector locking on.
	providers.SetPersonDownArrived(true)

	// Many dropouts well past clear_streak, but within arrivedRelease → stays locked.
	for i := 0; i < 50; i++ {
		s.classify(absent)
	}
	if !providers.PersonDownAlert() {
		t.Fatal("sticky lock: alert cleared during a dropout shorter than arrivedRelease")
	}
	if !providers.PersonDownArrived() {
		t.Fatal("sticky lock: arrived was released during a short dropout")
	}

	// A brief re-detection must restart the absence timer.
	s.classify(present)
	if s.clearStartedAt != (time.Time{}) {
		t.Fatal("re-detection should reset the absence timer")
	}

	// Now make the absence exceed arrivedRelease: prime one clear, then backdate it.
	s.classify(absent)
	s.clearStartedAt = time.Now().Add(-11 * time.Second)
	s.classify(absent)
	if providers.PersonDownAlert() {
		t.Fatal("sticky lock: alert should release after sustained absence > arrivedRelease")
	}
}

func TestNonArrivedClearsOnStreak(t *testing.T) {
	providers.SetPersonDownAlert(false)
	providers.SetPersonDownArrived(false)
	defer func() {
		providers.SetPersonDownAlert(false)
		providers.SetPersonDownArrived(false)
	}()

	s := newTestTracker(3, 0, 10*time.Second)
	present := providers.FallenSnapshot{Present: true, Alert: true, Name: "wendy", WidthFrac: 0.3}
	absent := providers.FallenSnapshot{Present: false}

	s.classify(present)
	// Not arrived: clears once the streak is met, no long release needed.
	s.classify(absent)
	s.classify(absent)
	if !providers.PersonDownAlert() {
		t.Fatal("should still hold before clear_streak is met")
	}
	s.classify(absent)
	if providers.PersonDownAlert() {
		t.Fatal("non-arrived alert should clear once clear_streak is met")
	}
}
