package providers

import (
	"sync"
	"sync/atomic"
)

// personDownAlert is set while a downed-person ALERT is latched.
var personDownAlert atomic.Bool

func SetPersonDownAlert(active bool) { personDownAlert.Store(active) }

func PersonDownAlert() bool { return personDownAlert.Load() }

// personDownArrived latches once the robot reaches the person (a "near" verdict), so
// motion code holds position until the alert clears.
var personDownArrived atomic.Bool

func SetPersonDownArrived(arrived bool) { personDownArrived.Store(arrived) }

func PersonDownArrived() bool { return personDownArrived.Load() }

// visionSeq increments once per fresh VLM verdict, so motion code can act at most once
// per perception frame.
var visionSeq atomic.Uint64

func BumpVisionSeq() { visionSeq.Add(1) }

func VisionSeq() uint64 { return visionSeq.Load() }

// FallenTarget is the geometry of the closest downed person, derived from a bbox
// detector and carried from the perception input to the motion connector so the
// robot can approach geometrically without the LLM in the loop.
type FallenTarget struct {
	Present    bool
	NormErrX   float64 // horizontal offset from frame center, [-1, 1]; + = right, - = left
	WidthFrac  float64 // bbox width as a fraction of frame width (distance proxy)
	Confidence float64
	Name       string
}

var (
	fallenTargetMu   sync.RWMutex
	fallenTargetData FallenTarget
)

// SetFallenTarget stores the latest downed-person geometry.
func SetFallenTarget(t FallenTarget) {
	fallenTargetMu.Lock()
	fallenTargetData = t
	fallenTargetMu.Unlock()
}

// FallenTargetSnapshot returns the latest downed-person geometry.
func FallenTargetSnapshot() FallenTarget {
	fallenTargetMu.RLock()
	defer fallenTargetMu.RUnlock()
	return fallenTargetData
}
