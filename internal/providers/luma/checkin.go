package luma

import (
	"sync/atomic"
	"time"
)

type CheckIn struct {
	Name string
	Time time.Time
}

var lastCheckIn atomic.Pointer[CheckIn]

func RecordCheckIn(name string, t time.Time) {
	lastCheckIn.Store(&CheckIn{Name: name, Time: t})
}

func LastCheckIn() *CheckIn {
	return lastCheckIn.Load()
}
