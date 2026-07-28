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

func TestVADLatencyTrackerRecordTranscriptPairsFIFOAndWritesJSONL(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "vad_asr_latency.jsonl")

	tr := &vadLatencyTracker{
		log:        zap.NewNop(),
		outputPath: outPath,
	}

	t0 := time.Now().Add(-2 * time.Second)
	t1 := t0.Add(500 * time.Millisecond)
	tr.pending = []time.Time{t0, t1}

	tr.recordTranscript("google", "first utterance")
	tr.recordTranscript("elevenlabs", "second utterance")

	if len(tr.pending) != 0 {
		t.Fatalf("expected pending queue drained, got %d remaining", len(tr.pending))
	}

	f, err := os.Open(outPath)
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
	if len(recs) != 2 {
		t.Fatalf("expected 2 records, got %d", len(recs))
	}

	if recs[0].Provider != "google" || recs[0].Transcript != "first utterance" {
		t.Errorf("unexpected first record: %+v", recs[0])
	}
	if recs[0].LatencyMS < 1900 || recs[0].LatencyMS > 2200 {
		t.Errorf("expected first latency near 2000ms, got %v", recs[0].LatencyMS)
	}
	if recs[1].Provider != "elevenlabs" || recs[1].Transcript != "second utterance" {
		t.Errorf("unexpected second record: %+v", recs[1])
	}
	if recs[1].LatencyMS <= recs[0].LatencyMS-2100 || recs[1].LatencyMS >= recs[0].LatencyMS {
		// second utterance's VAD-end (t1) is later than the first's (t0), and
		// both transcripts are recorded at roughly "now", so its latency
		// must be smaller than the first's by roughly 500ms.
		t.Errorf("expected second latency to be ~500ms less than first: first=%v second=%v", recs[0].LatencyMS, recs[1].LatencyMS)
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
