package qr_scanner_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/openmind/om1/internal/providers/luma"
	"github.com/openmind/om1/plugins/inputs/qr_scanner"
)

func TestFormatScanMessage_NoLumaConfigured(t *testing.T) {
	s := qr_scanner.NewTestSensor(nil, "", "", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-abc")
	want := "qr_scan: pk=g-1 event=evt-abc"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_EventMismatch(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-expected", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-other")
	want := "qr_scan_failed: pk=g-1 reason=event_mismatch"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_NotFound(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{Err: luma.ErrNotFound}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-abc", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-abc")
	want := "qr_scan_failed: pk=g-1 reason=guest_not_registered"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_Unauthorized(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{Err: luma.ErrUnauthorized}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-abc", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-abc")
	want := "qr_scan_failed: pk=g-1 reason=luma_auth"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_GenericError(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{Err: errors.New("network down")}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-abc", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-abc")
	want := "qr_scan_failed: pk=g-1 reason=lookup_error"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_EmptyResponse(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-abc", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-abc")
	want := "qr_scan_failed: pk=g-1 reason=empty_response"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_Success(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{Guest: &luma.Guest{
		UserFirstName: "Prachi",
		UserLastName:  "Singh",
		UserEmail:     "prachi@example.com",
	}}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}! Confirmed for {email}.", "evt-abc", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-abc")
	want := `qr_scan: name=Prachi greeting="Welcome, Prachi! Confirmed for prachi@example.com."`
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_SuccessStripsQuotesInGreeting(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{Guest: &luma.Guest{
		UserFirstName: "Ada",
		UserName:      `Ada "the original" Lovelace`,
	}}
	s := qr_scanner.NewTestSensor(lookup, `Welcome, {name}!`, "evt-abc", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "evt-abc")
	want := `qr_scan: name=Ada greeting="Welcome, Ada 'the original' Lovelace!"`
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestFormatScanMessage_EmptyEventIDSkipsMismatch(t *testing.T) {
	// When the QR doesn't carry an event id, we trust the configured expected one
	// and proceed with lookup rather than rejecting.
	lookup := &qr_scanner.FakeLookup{Guest: &luma.Guest{UserFirstName: "Sam"}}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-abc", time.Second)
	got := s.FormatScanMessage(context.Background(), "g-1", "")
	want := `qr_scan: name=Sam greeting="Welcome, Sam!"`
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

// fakeSpeaker captures AddText calls so tests can assert the direct-TTS path.
type fakeSpeaker struct{ texts []string }

func (f *fakeSpeaker) AddText(t string) { f.texts = append(f.texts, t) }

func TestFormatScanMessage_DirectTTSOnSuccess(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{Guest: &luma.Guest{
		UserFirstName: "Prachi",
		UserEmail:     "prachi@example.com",
	}}
	speaker := &fakeSpeaker{}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}! Confirmed for {email}.", "evt-abc", time.Second)
	s.SetSpeaker(speaker)

	_ = s.FormatScanMessage(context.Background(), "g-1", "evt-abc")

	if len(speaker.texts) != 1 {
		t.Fatalf("expected 1 AddText call, got %d", len(speaker.texts))
	}
	want := "Welcome, Prachi! Confirmed for prachi@example.com."
	if speaker.texts[0] != want {
		t.Fatalf("got %q want %q", speaker.texts[0], want)
	}
}

func TestFormatScanMessage_NoSpeakOnFailure(t *testing.T) {
	cases := []struct {
		name string
		err  error
	}{
		{"not_found", luma.ErrNotFound},
		{"unauthorized", luma.ErrUnauthorized},
		{"generic", errors.New("boom")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			lookup := &qr_scanner.FakeLookup{Err: tc.err}
			speaker := &fakeSpeaker{}
			s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-abc", time.Second)
			s.SetSpeaker(speaker)

			_ = s.FormatScanMessage(context.Background(), "g-1", "evt-abc")

			if len(speaker.texts) != 0 {
				t.Fatalf("expected no AddText calls on failure, got %v", speaker.texts)
			}
		})
	}
}

func TestFormatScanMessage_NoSpeakOnEventMismatch(t *testing.T) {
	lookup := &qr_scanner.FakeLookup{Guest: &luma.Guest{UserFirstName: "Sam"}}
	speaker := &fakeSpeaker{}
	s := qr_scanner.NewTestSensor(lookup, "Welcome, {first_name}!", "evt-abc", time.Second)
	s.SetSpeaker(speaker)

	_ = s.FormatScanMessage(context.Background(), "g-1", "evt-other")

	if len(speaker.texts) != 0 {
		t.Fatalf("expected no AddText on event mismatch, got %v", speaker.texts)
	}
}
