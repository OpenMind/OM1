package providers

import "sync/atomic"

// personDownAlert is set while a downed-person ALERT is latched, so motion code
// can soften turns during an approach.
var personDownAlert atomic.Bool

func SetPersonDownAlert(active bool) { personDownAlert.Store(active) }

func PersonDownAlert() bool { return personDownAlert.Load() }
