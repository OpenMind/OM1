package vad

import (
	"encoding/binary"
	"math"
	"testing"
	"time"
)

func TestObserveEmitsSpeechStartThenSpeechEnd(t *testing.T) {
	s := NewSegmenter(nil, SampleRate, SegmenterConfig{
		Threshold:          0.5,
		MinSilenceDuration: 300 * time.Millisecond,
	})

	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	if ev, ok := s.observe(0.9, base); !ok || ev.Type != EventSpeechStart {
		t.Fatalf("expected speech_start, got %+v ok=%v", ev, ok)
	}

	// Still speaking: no event.
	if _, ok := s.observe(0.8, base.Add(32*time.Millisecond)); ok {
		t.Fatalf("expected no event while still speaking")
	}

	silenceStart := base.Add(64 * time.Millisecond)
	if _, ok := s.observe(0.1, silenceStart); ok {
		t.Fatalf("expected no event on first silent frame (hangover not elapsed)")
	}

	// Silence hasn't persisted long enough yet.
	if _, ok := s.observe(0.1, silenceStart.Add(100*time.Millisecond)); ok {
		t.Fatalf("expected no event before MinSilenceDuration elapses")
	}

	// Now past the hangover window.
	confirmAt := silenceStart.Add(300 * time.Millisecond)
	ev, ok := s.observe(0.1, confirmAt)
	if !ok || ev.Type != EventSpeechEnd {
		t.Fatalf("expected speech_end, got %+v ok=%v", ev, ok)
	}
	if !ev.At.Equal(silenceStart) {
		t.Errorf("speech_end should be timestamped at the true silence onset %v, got %v", silenceStart, ev.At)
	}
}

func TestObserveIgnoresBriefDips(t *testing.T) {
	s := NewSegmenter(nil, SampleRate, SegmenterConfig{
		Threshold:          0.5,
		MinSilenceDuration: 300 * time.Millisecond,
	})
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	if _, ok := s.observe(0.9, base); !ok {
		t.Fatalf("expected speech_start")
	}
	// A brief dip below threshold that recovers before the hangover elapses
	// must not produce speech_end, and speech continues.
	if _, ok := s.observe(0.1, base.Add(50*time.Millisecond)); ok {
		t.Fatalf("brief dip should not emit an event yet")
	}
	if _, ok := s.observe(0.9, base.Add(80*time.Millisecond)); ok {
		t.Fatalf("recovering above threshold should not emit an event")
	}
	if !s.inSpeech {
		t.Fatalf("segmenter should still consider itself in speech after recovering")
	}
}

func TestObserveNoSpeechEndWithoutPriorSpeechStart(t *testing.T) {
	s := NewSegmenter(nil, SampleRate, SegmenterConfig{})
	if _, ok := s.observe(0.1, time.Now()); ok {
		t.Fatalf("silence with no prior speech should never emit an event")
	}
}

func TestResampleSameRateIsPassthrough(t *testing.T) {
	s := NewSegmenter(nil, SampleRate, SegmenterConfig{})
	in := []int16{0, 16384, -16384, 32767}
	out := s.resample(in)
	if len(out) != len(in) {
		t.Fatalf("expected %d samples, got %d", len(in), len(out))
	}
	if math.Abs(float64(out[1])-0.5) > 1e-4 {
		t.Errorf("expected ~0.5, got %v", out[1])
	}
}

func TestResampleDownsamplesAndIsContinuousAcrossCalls(t *testing.T) {
	// 48000 -> 16000 is an exact 3:1 ratio.
	full := NewSegmenter(nil, 48000, SegmenterConfig{})
	src := make([]int16, 300)
	for i := range src {
		src[i] = int16(i * 100)
	}
	allAtOnce := full.resample(src)

	split := NewSegmenter(nil, 48000, SegmenterConfig{})
	var piecewise []float32
	for i := 0; i < len(src); i += 37 { // odd chunk size to stress the carry-over path
		end := i + 37
		if end > len(src) {
			end = len(src)
		}
		piecewise = append(piecewise, split.resample(src[i:end])...)
	}

	// Chunking can shift exactly where the last fractional sample resolves by
	// one output sample; the overlapping prefix must still match exactly.
	if diff := len(allAtOnce) - len(piecewise); diff < -1 || diff > 1 {
		t.Fatalf("length mismatch: all-at-once=%d piecewise=%d", len(allAtOnce), len(piecewise))
	}
	n := len(allAtOnce)
	if len(piecewise) < n {
		n = len(piecewise)
	}
	for i := 0; i < n; i++ {
		if math.Abs(float64(allAtOnce[i]-piecewise[i])) > 1e-5 {
			t.Fatalf("sample %d differs: all-at-once=%v piecewise=%v", i, allAtOnce[i], piecewise[i])
		}
	}
	if len(allAtOnce) < 90 || len(allAtOnce) > 100 {
		t.Errorf("expected roughly 100 output samples for 300 input samples at 3:1, got %d", len(allAtOnce))
	}
}

func TestBytesToInt16(t *testing.T) {
	buf := make([]byte, 4)
	var neg int16 = -1
	binary.LittleEndian.PutUint16(buf[0:], uint16(neg))
	binary.LittleEndian.PutUint16(buf[2:], uint16(1234))
	out := bytesToInt16(buf)
	if len(out) != 2 || out[0] != -1 || out[1] != 1234 {
		t.Fatalf("unexpected decode: %v", out)
	}
}
