package asr

import (
	"testing"
	"time"
)

// The window handed to active-speaker detection must cover the talking and
// nothing else. These pin the three ways it gets its end.

func TestWindowEndsAtTheVendorsSpeechEnd(t *testing.T) {
	s := &transcriberStream{}
	s.speechStartTime = time.Now().Add(-2 * time.Second)
	s.speechEndTime = time.Now().Add(-800 * time.Millisecond)

	start, end := s.speechWindow()

	if !end.Equal(s.speechEndTime) {
		t.Fatalf("want the recorded speech end %v, got %v", s.speechEndTime, end)
	}
	if spoken := end.Sub(start); spoken > 1300*time.Millisecond {
		t.Errorf("window ran %v; the endpointing delay is being counted as speech", spoken)
	}
}

func TestWindowFallsBackToNowWithoutASpeechEnd(t *testing.T) {
	// ElevenLabs' path: the transcript itself marks the end.
	s := &transcriberStream{}
	s.speechStartTime = time.Now().Add(-time.Second)

	_, end := s.speechWindow()

	if time.Since(end) > 50*time.Millisecond {
		t.Errorf("fallback end should be now, was %v ago", time.Since(end))
	}
}

func TestStaleSpeechEndIsIgnored(t *testing.T) {
	// A mark left over from the previous utterance would end this window
	// before it began, and /speaking would be asked about negative time.
	s := &transcriberStream{}
	s.speechEndTime = time.Now().Add(-10 * time.Second)
	s.speechStartTime = time.Now().Add(-time.Second)

	start, end := s.speechWindow()

	if !end.After(start) {
		t.Fatalf("window ends before it starts: %v -> %v", start, end)
	}
	if time.Since(end) > 50*time.Millisecond {
		t.Errorf("should have fallen back to now, was %v ago", time.Since(end))
	}
}
