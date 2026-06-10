package qr_scanner

import "time"

var (
	ParseLumaCheckinURL = parseLumaCheckinURL
	DecodeQR            = decodeQR
	ErrQRNotFound       = errQRNotFound
)

type Debouncer = debouncer

func NewDebouncer(window time.Duration) *Debouncer { return newDebouncer(window) }

func (d *Debouncer) SetNow(f func() time.Time) { d.now = f }
func (d *Debouncer) Has(key string) bool       { _, ok := d.seen[key]; return ok }
