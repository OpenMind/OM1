package autonomy

import (
	"testing"

	"github.com/openmind/om1/internal/providers"
)

func TestApproachDecision(t *testing.T) {
	const (
		lock = providers.DefaultLockWidthFrac // 0.7
		tol  = providers.DefaultCenterTol     // 0.33
	)

	tests := []struct {
		name      string
		normErrX  float64
		widthFrac float64
		want      approachAction
	}{
		{"near and centered -> lock", 0.0, 0.8, approachLock},
		{"near overrides off-center -> lock", 0.9, 0.75, approachLock},
		{"far and centered -> advance", 0.1, 0.3, approachAdvance},
		{"far and right -> recenter right", 0.5, 0.3, approachRecenterRight},
		{"far and left -> recenter left", -0.5, 0.3, approachRecenterLeft},
		{"just inside tolerance -> advance", 0.33, 0.3, approachAdvance},
		{"just at lock threshold -> lock", 0.0, 0.7, approachLock},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := approachDecision(tt.normErrX, tt.widthFrac, lock, tol); got != tt.want {
				t.Errorf("approachDecision(%v, %v) = %v, want %v",
					tt.normErrX, tt.widthFrac, got, tt.want)
			}
		})
	}
}
