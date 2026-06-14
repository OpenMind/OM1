package qr_scanner

import (
	"context"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers/luma"
)

var (
	ParseLumaCheckinURL = parseLumaCheckinURL
	DecodeQR            = decodeQR
	ErrQRNotFound       = errQRNotFound
)

type Debouncer = debouncer

func NewDebouncer(window time.Duration) *Debouncer { return newDebouncer(window) }

func (d *Debouncer) SetNow(f func() time.Time) { d.now = f }
func (d *Debouncer) Has(key string) bool       { _, ok := d.seen[key]; return ok }

// GuestLookup is exported for tests so they can inject a fake without touching
// the http stack.
type GuestLookup = guestLookup

// NewTestSensor builds a sensor wired with the given lookup, greeting template
// and expected event id. The frame source and debouncer are not used by tests
// that only exercise formatScanMessage.
func NewTestSensor(lookup GuestLookup, greetingTmpl, expectedEventID string, timeout time.Duration) *TestSensor {
	return &TestSensor{&sensor{
		log:             zap.NewNop(),
		luma:            lookup,
		greetingTmpl:    greetingTmpl,
		lumaTimeout:     timeout,
		expectedEventID: expectedEventID,
	}}
}

type TestSensor struct{ *sensor }

func (s *TestSensor) FormatScanMessage(ctx context.Context, pk, eventID string) string {
	return s.formatScanMessage(ctx, pk, eventID)
}

// SetSpeaker injects a fake ttsSpeaker so tests can verify the direct-TTS
// path without touching the real ElevenLabs singleton.
func (s *TestSensor) SetSpeaker(sp Speaker) { s.speak = sp }

// Speaker is the test-side alias for the unexported ttsSpeaker interface.
type Speaker = ttsSpeaker

// FakeLookup implements GuestLookup for tests.
type FakeLookup struct {
	Guest *luma.Guest
	Err   error
}

func (f *FakeLookup) GetGuest(_ context.Context, _ string) (*luma.Guest, error) {
	return f.Guest, f.Err
}

func (f *FakeLookup) CheckIn(_ context.Context, _ *luma.Guest) error {
	return nil
}
