package asr

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"go.uber.org/zap"

	"github.com/openmind/om1/internal/providers/tts"
)

func resetTTSState(t *testing.T) {
	t.Helper()
	tts.Speaking.Store(false)
	tts.Interrupt.Store(false)
	t.Cleanup(func() {
		tts.Speaking.Store(false)
		tts.Interrupt.Store(false)
	})
}

func TestVADLatencyTrackerNilIsSafe(t *testing.T) {
	var tr *vadLatencyTracker
	tr.feedAudio([]byte{1, 2, 3, 4})
	tr.recordTranscript("google", "english", "v2", "hello there")
	tr.close()
}

func TestNewVADLatencyTrackerDisabledReturnsNil(t *testing.T) {
	tr := newVADLatencyTracker(vadLatencyConfig{EnableVADLatency: false}, false, 16000, zap.NewNop())
	if tr != nil {
		t.Fatalf("expected nil tracker when disabled, got %+v", tr)
	}
}

func TestNewVADLatencyTrackerDegradesGracefullyWithoutRuntime(t *testing.T) {
	tr := newVADLatencyTracker(vadLatencyConfig{
		EnableVADLatency: true,
		VADModelPath:     "/nonexistent/model.onnx",
		VADLibraryPath:   "/nonexistent/libonnxruntime.so",
	}, false, 16000, zap.NewNop())
	if tr != nil {
		t.Fatalf("expected nil tracker when the onnxruntime library is unavailable, got %+v", tr)
	}
}

func TestNewVADLatencyTrackerInterruptOnlyAlsoAttemptsLoad(t *testing.T) {
	tr := newVADLatencyTracker(vadLatencyConfig{
		EnableVADLatency: false,
		VADModelPath:     "/nonexistent/model.onnx",
		VADLibraryPath:   "/nonexistent/libonnxruntime.so",
	}, true, 16000, zap.NewNop())
	if tr != nil {
		t.Fatalf("expected nil tracker when the onnxruntime library is unavailable, got %+v", tr)
	}
}

func readVADLatencyRecords(t *testing.T, path string) []vadLatencyRecord {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open output: %v", err)
	}
	defer func() { _ = f.Close() }()

	var recs []vadLatencyRecord
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var rec vadLatencyRecord
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			t.Fatalf("unmarshal record: %v", err)
		}
		recs = append(recs, rec)
	}
	return recs
}

func TestVADLatencyTrackerRecordTranscriptPairsAndWritesJSONL(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "vad_asr_latency.jsonl")

	tr := &vadLatencyTracker{
		log:        zap.NewNop(),
		outputPath: outPath,
	}

	vadEnd := time.Now().Add(-2 * time.Second)
	tr.pendingEnd = vadEnd
	tr.recordTranscript("google", "english", "v2", "first utterance")

	if !tr.pendingEnd.IsZero() {
		t.Fatalf("expected pendingEnd cleared after being consumed, got %v", tr.pendingEnd)
	}

	recs := readVADLatencyRecords(t, outPath)
	if len(recs) != 1 {
		t.Fatalf("expected 1 record, got %d", len(recs))
	}
	if recs[0].Provider != "google" || recs[0].Transcript != "first utterance" {
		t.Errorf("unexpected record: %+v", recs[0])
	}
	if recs[0].LatencyMS < 1900 || recs[0].LatencyMS > 2200 {
		t.Errorf("expected latency near 2000ms, got %v", recs[0].LatencyMS)
	}
}

func TestVADLatencyTrackerDiscardsStaleUnmatchedEvent(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "vad_asr_latency.jsonl")

	tr := &vadLatencyTracker{
		log:        zap.NewNop(),
		outputPath: outPath,
	}

	tr.pendingEnd = time.Now().Add(-58 * time.Second) // e.g. a noise blip ASR never transcribed
	tr.pendingEnd = time.Now().Add(-3 * time.Second)  // the real end-of-speech

	tr.recordTranscript("google", "english", "v2", "what are you doing this afternoon")

	recs := readVADLatencyRecords(t, outPath)
	if len(recs) != 1 {
		t.Fatalf("expected 1 record, got %d", len(recs))
	}
	if recs[0].LatencyMS < 2900 || recs[0].LatencyMS > 3200 {
		t.Errorf("expected latency near 3000ms (paired with the recent event, not the 58s-old stale one), got %v", recs[0].LatencyMS)
	}
}

func TestVADLatencyTrackerDiscardsImplausiblyOldPairing(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "vad_asr_latency.jsonl")

	tr := &vadLatencyTracker{
		log:        zap.NewNop(),
		outputPath: outPath,
	}

	tr.pendingEnd = time.Now().Add(-maxSanePendingAge - time.Second)
	tr.recordTranscript("google", "english", "v2", "should be discarded as implausible")

	if _, err := os.Stat(outPath); !os.IsNotExist(err) {
		t.Fatalf("expected no output file for an implausibly old pairing, stat err=%v", err)
	}
}

func TestVADLatencyTrackerRecordTranscriptNoPendingIsNoop(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "vad_asr_latency.jsonl")

	tr := &vadLatencyTracker{
		log:        zap.NewNop(),
		outputPath: outPath,
	}
	tr.recordTranscript("google", "english", "v2", "unexpected transcript")

	if _, err := os.Stat(outPath); !os.IsNotExist(err) {
		t.Fatalf("expected no output file to be created, stat err=%v", err)
	}
}

func TestCheckInterruptFiresAfterConfirmDelayWhileTTSSpeaking(t *testing.T) {
	resetTTSState(t)
	tts.Speaking.Store(true)

	now := time.Now()
	tr := &vadLatencyTracker{
		log:                   zap.NewNop(),
		enableInterrupt:       true,
		interruptConfirmDelay: 150 * time.Millisecond,
		speechActive:          true,
		candidateStart:        now.Add(-200 * time.Millisecond),
	}

	tr.checkInterrupt(now)

	if !tr.confirmed {
		t.Error("expected candidate to be marked confirmed")
	}
	if !tts.Interrupt.Load() {
		t.Error("expected tts.RequestInterrupt to have fired")
	}
}

func TestCheckInterruptIgnoresBlipShorterThanConfirmDelay(t *testing.T) {
	resetTTSState(t)
	tts.Speaking.Store(true)

	now := time.Now()
	tr := &vadLatencyTracker{
		log:                   zap.NewNop(),
		enableInterrupt:       true,
		interruptConfirmDelay: 150 * time.Millisecond,
		speechActive:          true,
		candidateStart:        now.Add(-50 * time.Millisecond), // shorter than the 150ms confirm delay
	}

	tr.checkInterrupt(now)

	if tr.confirmed {
		t.Error("expected a sub-confirm-delay blip to not be confirmed yet")
	}
	if tts.Interrupt.Load() {
		t.Error("expected tts.RequestInterrupt to not have fired for a blip")
	}
}

func TestCheckInterruptSkipsRequestWhenTTSNotSpeaking(t *testing.T) {
	resetTTSState(t)
	tts.Speaking.Store(false)

	now := time.Now()
	tr := &vadLatencyTracker{
		log:                   zap.NewNop(),
		enableInterrupt:       true,
		interruptConfirmDelay: 150 * time.Millisecond,
		speechActive:          true,
		candidateStart:        now.Add(-200 * time.Millisecond),
	}

	tr.checkInterrupt(now)

	if !tr.confirmed {
		t.Error("expected candidate to be marked confirmed even with nothing to interrupt")
	}
	if tts.Interrupt.Load() {
		t.Error("expected tts.RequestInterrupt to not fire when TTS isn't speaking")
	}
}

func TestCheckInterruptNoopWhenInterruptDisabled(t *testing.T) {
	resetTTSState(t)
	tts.Speaking.Store(true)

	now := time.Now()
	tr := &vadLatencyTracker{
		log:                   zap.NewNop(),
		enableInterrupt:       false,
		interruptConfirmDelay: 150 * time.Millisecond,
		speechActive:          true,
		candidateStart:        now.Add(-200 * time.Millisecond),
	}

	tr.checkInterrupt(now)

	if tr.confirmed {
		t.Error("expected no-op when interrupt tracking is disabled")
	}
	if tts.Interrupt.Load() {
		t.Error("expected tts.RequestInterrupt to not fire when interrupt tracking is disabled")
	}
}

func TestCheckInterruptDoesNotRefireOnceConfirmed(t *testing.T) {
	resetTTSState(t)
	tts.Speaking.Store(true)

	now := time.Now()
	tr := &vadLatencyTracker{
		log:                   zap.NewNop(),
		enableInterrupt:       true,
		interruptConfirmDelay: 150 * time.Millisecond,
		speechActive:          true,
		candidateStart:        now.Add(-200 * time.Millisecond),
		confirmed:             true,
	}

	tr.checkInterrupt(now)

	if tts.Interrupt.Load() {
		t.Error("expected tts.RequestInterrupt to not fire again once already confirmed")
	}
}
