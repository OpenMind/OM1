package providers

import "sync/atomic"

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
