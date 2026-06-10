package qr_scanner_test

import (
	"testing"
	"time"

	"github.com/openmind/om1/plugins/inputs/qr_scanner"
)

func TestDebouncerAcceptsThenRejects(t *testing.T) {
	d := qr_scanner.NewDebouncer(30 * time.Second)
	now := time.Unix(1_700_000_000, 0)
	d.SetNow(func() time.Time { return now })

	if !d.TryRecord("g-1") {
		t.Fatalf("first record should be accepted")
	}
	if d.TryRecord("g-1") {
		t.Fatalf("immediate re-record should be rejected")
	}

	now = now.Add(29 * time.Second)
	if d.TryRecord("g-1") {
		t.Fatalf("re-record within window should be rejected")
	}

	now = now.Add(2 * time.Second) // total 31s after first
	if !d.TryRecord("g-1") {
		t.Fatalf("re-record after window should be accepted")
	}
}

func TestDebouncerDistinctKeysIndependent(t *testing.T) {
	d := qr_scanner.NewDebouncer(30 * time.Second)
	if !d.TryRecord("g-1") {
		t.Fatalf("g-1 first should be accepted")
	}
	if !d.TryRecord("g-2") {
		t.Fatalf("g-2 first should be accepted")
	}
}

func TestDebouncerPrunesOldEntries(t *testing.T) {
	d := qr_scanner.NewDebouncer(1 * time.Second)
	now := time.Unix(1_700_000_000, 0)
	d.SetNow(func() time.Time { return now })

	d.TryRecord("g-old")
	if !d.Has("g-old") {
		t.Fatalf("expected g-old to be recorded")
	}

	now = now.Add(15 * time.Second) // 15x window
	d.TryRecord("g-fresh")

	if d.Has("g-old") {
		t.Fatalf("expected g-old to be pruned after 15x window")
	}
	if !d.Has("g-fresh") {
		t.Fatalf("expected g-fresh to be present")
	}
}
