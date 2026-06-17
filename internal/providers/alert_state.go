package providers

import "sync/atomic"

// personDownAlert is set while a downed-person ALERT is latched, so motion code
// can soften turns during an approach.
var personDownAlert atomic.Bool

func SetPersonDownAlert(active bool) { personDownAlert.Store(active) }

func PersonDownAlert() bool { return personDownAlert.Load() }

// personDownArrived latches once the robot has reached and centered on a downed
// person (a 'Distance: near' + centered verdict). While set, motion code holds
// position instead of re-centering, because the VLM Location is jittery and chasing
// it over-rotates and loses the person. Cleared when the alert ends.
var personDownArrived atomic.Bool

func SetPersonDownArrived(arrived bool) { personDownArrived.Store(arrived) }

func PersonDownArrived() bool { return personDownArrived.Load() }

// visionSeq increments once per fresh VLM verdict. The cortex re-serves the latched
// verdict every tick (~1 Hz) while the VLM only refreshes every few seconds, so motion
// code reads this to act at most once per perception frame.
var visionSeq atomic.Uint64

// BumpVisionSeq marks that a new VLM verdict has replaced the latched one.
func BumpVisionSeq() { visionSeq.Add(1) }

// VisionSeq returns the current perception-frame counter.
func VisionSeq() uint64 { return visionSeq.Load() }
