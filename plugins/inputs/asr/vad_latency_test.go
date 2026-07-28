package asr

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestVADLatencyTrackerNilIsSafe(t *testing.T) {
	var tr *vadLatencyTracker
	tr.feedAudio([]byte{1, 2, 3, 4})
	tr.recordTranscript("google", "hello there")
	tr.close()
}

func TestNewVADLatencyTrackerDisabledReturnsNil(t *testing.T) {
	tr := newVADLatencyTracker(vadLatencyConfig{EnableVADLatency: false}, 16000, zap.NewNop())
	if tr != nil {
		t.Fatalf("expected nil tracker when disabled, got %+v", tr)
	}
}

func TestNewVADLatencyTrackerDegradesGracefullyWithoutRuntime(t *testing.T) {
	// No onnxruntime shared library is installed in the test environment, so
	// loading must fail cleanly and the tracker must come back nil rather
	// than panicking or blocking ASR startup.
	tr := newVADLatencyTracker(vadLatencyConfig{
		EnableVADLatency: true,
		VADModelPath:     "/nonexistent/model.onnx",
		VADLibraryPath:   "/nonexistent/libonnxruntime.so",
	}, 16000, zap.NewNop())
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
	tr.recordTranscript("google", "first utterance")

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

// TestVADLatencyTrackerDiscardsStaleUnmatchedEvent covers the bug where a
// VAD speech_end that never got a matching transcript (e.g. a short blip
// acceptASRTranscript rejected) would sit in a FIFO queue and later pair
// with a much later, unrelated transcript, producing a nonsensical latency.
// A newer speech_end must fully replace the stale one, not queue behind it.
func TestVADLatencyTrackerDiscardsStaleUnmatchedEvent(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "vad_asr_latency.jsonl")

	tr := &vadLatencyTracker{
		log:        zap.NewNop(),
		outputPath: outPath,
	}

	// Simulates what feedAudio does on each new speech_end: unconditionally
	// overwrite pendingEnd, exactly as if a stale event were followed by a
	// real one before any transcript consumed the stale one.
	tr.pendingEnd = time.Now().Add(-58 * time.Second) // e.g. a noise blip ASR never transcribed
	tr.pendingEnd = time.Now().Add(-3 * time.Second)  // the real end-of-speech

	tr.recordTranscript("google", "what are you doing this afternoon")

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
	tr.recordTranscript("google", "should be discarded as implausible")

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
	tr.recordTranscript("google", "unexpected transcript")

	if _, err := os.Stat(outPath); !os.IsNotExist(err) {
		t.Fatalf("expected no output file to be created, stat err=%v", err)
	}
}
