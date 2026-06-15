package luma

import (
	"testing"
	"time"
)

func TestRecordAndLastCheckIn(t *testing.T) {
	if got := LastCheckIn(); got != nil {
		t.Fatalf("expected nil before any publish, got %+v", got)
	}

	t1 := time.Unix(1_700_000_000, 0)
	RecordCheckIn("Ada", t1)

	got := LastCheckIn()
	if got == nil {
		t.Fatal("expected a check-in after publish")
	}
	if got.Name != "Ada" || !got.Time.Equal(t1) {
		t.Fatalf("unexpected check-in: %+v", got)
	}

	// A later publish replaces the previous value.
	t2 := t1.Add(time.Minute)
	RecordCheckIn("Grace", t2)
	got = LastCheckIn()
	if got.Name != "Grace" || !got.Time.Equal(t2) {
		t.Fatalf("expected latest check-in to be Grace@t2, got %+v", got)
	}
}
