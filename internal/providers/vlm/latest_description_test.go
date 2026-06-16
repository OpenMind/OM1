package vlm

import (
	"testing"
	"time"
)

func TestLatestDescriptionSetGet(t *testing.T) {
	p := &LatestDescriptionProvider{}

	if _, _, ok := p.Get(); ok {
		t.Fatal("expected no description before any Set")
	}

	ts := time.Now()
	p.Set("a person waving", ts)

	got, gotTS, ok := p.Get()
	if !ok {
		t.Fatal("expected a description after Set")
	}
	if got != "a person waving" {
		t.Fatalf("unexpected description: %q", got)
	}
	if !gotTS.Equal(ts) {
		t.Fatalf("unexpected timestamp: %v != %v", gotTS, ts)
	}
}

func TestLatestDescriptionSetIgnoresEmpty(t *testing.T) {
	p := &LatestDescriptionProvider{}
	p.Set("", time.Now())
	if _, _, ok := p.Get(); ok {
		t.Fatal("empty description should be ignored")
	}
}

func TestLatestDescriptionGetFreshStaleness(t *testing.T) {
	p := &LatestDescriptionProvider{}
	p.Set("stale scene", time.Now().Add(-time.Minute))

	if _, _, ok := p.GetFresh(time.Second); ok {
		t.Fatal("expected stale description to be rejected")
	}
	if _, _, ok := p.GetFresh(0); !ok {
		t.Fatal("non-positive maxAge should disable staleness check")
	}
	if _, _, ok := p.GetFresh(2 * time.Minute); !ok {
		t.Fatal("description within maxAge should be returned")
	}
}
