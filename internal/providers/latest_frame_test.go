package providers

import (
	"context"
	"testing"
	"time"
)

func TestLatestFrameSetGet(t *testing.T) {
	p := &LatestFrameProvider{}

	if _, _, ok := p.Get(); ok {
		t.Fatal("expected no frame before any Set")
	}

	ts := time.Now()
	p.Set([]byte{0x01, 0x02}, ts)

	got, gotTS, ok := p.Get()
	if !ok {
		t.Fatal("expected a frame after Set")
	}
	if string(got) != string([]byte{0x01, 0x02}) {
		t.Fatalf("unexpected frame bytes: %v", got)
	}
	if !gotTS.Equal(ts) {
		t.Fatalf("unexpected timestamp: %v != %v", gotTS, ts)
	}
}

func TestLatestFrameCopiesOnSetAndGet(t *testing.T) {
	p := &LatestFrameProvider{}

	src := []byte{0x01, 0x02}
	p.Set(src, time.Now())
	src[0] = 0xFF

	got, _, _ := p.Get()
	if got[0] != 0x01 {
		t.Fatalf("stored frame aliases caller slice: %v", got)
	}

	got[1] = 0xFF
	again, _, _ := p.Get()
	if again[1] != 0x02 {
		t.Fatalf("returned frame aliases stored slice: %v", again)
	}
}

func TestLatestFrameSetIgnoresEmpty(t *testing.T) {
	p := &LatestFrameProvider{}
	p.Set(nil, time.Now())
	if _, _, ok := p.Get(); ok {
		t.Fatal("empty frame should be ignored")
	}
}

func TestLatestFrameGetFreshStaleness(t *testing.T) {
	p := &LatestFrameProvider{}
	p.Set([]byte{0x01}, time.Now().Add(-time.Minute))

	if _, _, ok := p.GetFresh(time.Second); ok {
		t.Fatal("expected stale frame to be rejected")
	}
	if _, _, ok := p.GetFresh(0); !ok {
		t.Fatal("non-positive maxAge should disable staleness check")
	}
	if _, _, ok := p.GetFresh(2 * time.Minute); !ok {
		t.Fatal("frame within maxAge should be returned")
	}
}

func TestLatestFrameWaitForFreshImmediate(t *testing.T) {
	p := &LatestFrameProvider{}
	p.Set([]byte{0x01}, time.Now())

	got, _, ok := p.WaitForFresh(context.Background(), time.Second, time.Second)
	if !ok || len(got) != 1 {
		t.Fatalf("expected an already-fresh frame to return immediately, ok=%v got=%v", ok, got)
	}
}

func TestLatestFrameWaitForFreshArrives(t *testing.T) {
	p := &LatestFrameProvider{}

	go func() {
		time.Sleep(20 * time.Millisecond)
		p.Set([]byte{0x07}, time.Now())
	}()

	got, _, ok := p.WaitForFresh(context.Background(), time.Second, time.Second)
	if !ok || len(got) != 1 || got[0] != 0x07 {
		t.Fatalf("expected to receive the frame that arrived during the wait, ok=%v got=%v", ok, got)
	}
}

func TestLatestFrameWaitForFreshTimeout(t *testing.T) {
	p := &LatestFrameProvider{}

	start := time.Now()
	if _, _, ok := p.WaitForFresh(context.Background(), time.Second, 30*time.Millisecond); ok {
		t.Fatal("expected timeout when no frame arrives")
	}
	if elapsed := time.Since(start); elapsed < 30*time.Millisecond {
		t.Fatalf("returned before timeout elapsed: %v", elapsed)
	}
}

func TestLatestFrameWaitForFreshNoWait(t *testing.T) {
	p := &LatestFrameProvider{}
	if _, _, ok := p.WaitForFresh(context.Background(), time.Second, 0); ok {
		t.Fatal("non-positive timeout should not block and should report no frame")
	}
}

func TestLatestFrameWaitForFreshContextCancel(t *testing.T) {
	p := &LatestFrameProvider{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, _, ok := p.WaitForFresh(ctx, time.Second, time.Minute); ok {
		t.Fatal("expected no frame when context is already cancelled")
	}
}
