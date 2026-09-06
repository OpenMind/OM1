package vlm

import (
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/inputs"
	"github.com/openmind/om1/internal/providers"
)

func newTestDetector(clearStreak int, minHold time.Duration) *laydownDetector {
	return &laydownDetector{
		name:        "PersonLaydownDetector",
		log:         zap.NewNop(),
		clearStreak: clearStreak,
		minHold:     minHold,
	}
}

func resetAlertState() {
	providers.SetPersonDownAlert(false)
	providers.SetPersonDownArrived(false)
}

func TestClassifyAlertSetsAlert(t *testing.T) {
	resetAlertState()
	d := newTestDetector(2, 0)

	text := "ALERT: a person is lying on the ground. Location in view: center. Distance: far."
	got := d.classify(text)

	assert.Equal(t, text, got)
	assert.True(t, d.alerted)
	assert.True(t, providers.PersonDownAlert())
}

// A non-alert verdict whose description merely contains the word "alert" must not
// enter emergency mode: detection keys on the ALERT prefix, not a substring.
func TestClassifyAlertWordInDescriptionIsNotAlert(t *testing.T) {
	resetAlertState()
	d := newTestDetector(2, 0)

	text := "No person lying on the ground. I see a person standing and looking alert."
	got := d.classify(text)

	assert.Equal(t, text, got)
	assert.False(t, d.alerted)
	assert.False(t, providers.PersonDownAlert())
}

func TestClassifyNearLatchesArrivedAndHolds(t *testing.T) {
	resetAlertState()
	d := newTestDetector(2, 0)

	d.classify("ALERT: a person is lying on the ground. Location in view: center. Distance: far.")
	assert.True(t, providers.PersonDownAlert())
	assert.False(t, providers.PersonDownArrived(), "far should not latch arrival")

	d.classify("ALERT: a person is lying on the ground. Location in view: left. Distance: near.")
	assert.True(t, providers.PersonDownArrived(), "near should latch arrival")

	// A later far/off-center verdict must not unlatch arrival.
	d.classify("ALERT: a person is lying on the ground. Location in view: right. Distance: far.")
	assert.True(t, providers.PersonDownArrived(), "arrival stays latched until the alert clears")
}

func TestClassifyClearRequiresStreak(t *testing.T) {
	resetAlertState()
	d := newTestDetector(2, 0)

	alert := "ALERT: a person is lying on the ground. Location in view: center. Distance: near."
	d.classify(alert)
	assert.True(t, providers.PersonDownArrived())

	// First non-alert verdict: streak not met yet, alert is held and re-served.
	got := d.classify("No person lying on the ground. I see a room.")
	assert.Equal(t, alert, got)
	assert.True(t, providers.PersonDownAlert())

	// Second consecutive non-alert verdict meets the streak: alert clears.
	cleared := "No person lying on the ground. I see a room."
	got = d.classify(cleared)
	assert.Equal(t, cleared, got)
	assert.False(t, providers.PersonDownAlert())
	assert.False(t, providers.PersonDownArrived())
}

func TestClassifyMinHoldBlocksClear(t *testing.T) {
	resetAlertState()
	d := newTestDetector(1, time.Hour)

	d.classify("ALERT: a person is lying on the ground. Location in view: center. Distance: near.")

	// Streak is met (1) but the min-hold window has not elapsed, so the alert holds.
	got := d.classify("No person lying on the ground. I see a room.")
	assert.Contains(t, got, "ALERT")
	assert.True(t, providers.PersonDownAlert())
}

func TestFormattedLatestBuffer(t *testing.T) {
	d := newTestDetector(2, 0)

	assert.Equal(t, "", d.FormattedLatestBuffer(), "no verdict yet")

	d.latest = *inputs.NewMessage("ALERT: a person is lying on the ground.")
	d.hasLatest = true

	out := d.FormattedLatestBuffer()
	assert.True(t, strings.Contains(out, vlmDescriptor), "should carry the Vision prefix")
	assert.True(t, strings.Contains(out, "ALERT: a person is lying on the ground."))
}
