package traceexport

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/openmind/om1/internal/metrics"
	"github.com/openmind/om1/internal/tracer/tracetype"
)

// waitFor polls fn until it returns true or the deadline passes.
func waitFor(t *testing.T, timeout time.Duration, fn func() bool) {
	t.Helper()
	deadline := time.After(timeout)
	for {
		if fn() {
			return
		}
		select {
		case <-deadline:
			t.Fatal("timed out waiting for condition")
		case <-time.After(5 * time.Millisecond):
		}
	}
}

func TestStart_exportsRecordAsGaugeSeries(t *testing.T) {
	records := make(chan tracetype.TraceRecord, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	Start(ctx, records, zap.NewNop())

	records <- tracetype.TraceRecord{
		Timestamp:  "2026-08-31T17:41:36.027193146Z",
		Generation: 1,
		LLMInput:   "hello there",
		LLMOutput:  []map[string]any{{"type": "speak", "value": "hi"}},
	}

	waitFor(t, 2*time.Second, func() bool {
		g := metrics.TraceInfo.WithLabelValues("0", "2026-08-31T17:41:36.027193146Z", "1",
			"hello there", `[{"type":"speak","value":"hi"}]`)
		return testutil.ToFloat64(g) == 1
	})
}

func TestStart_evictsOldestBeyondMaxBuffered(t *testing.T) {
	records := make(chan tracetype.TraceRecord, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	Start(ctx, records, zap.NewNop())

	for i := 0; i <= maxBuffered; i++ {
		records <- tracetype.TraceRecord{
			Timestamp:  fmt.Sprintf("ts-%d", i),
			Generation: 1,
			LLMInput:   "input",
			LLMOutput:  []map[string]any{},
		}
	}

	lastSeq := fmt.Sprintf("%d", maxBuffered)
	waitFor(t, 2*time.Second, func() bool {
		g := metrics.TraceInfo.WithLabelValues(lastSeq, fmt.Sprintf("ts-%d", maxBuffered), "1", "input", "[]")
		return testutil.ToFloat64(g) == 1
	})

	evicted := metrics.TraceInfo.WithLabelValues("0", "ts-0", "1", "input", "[]")
	require.Equal(t, 0.0, testutil.ToFloat64(evicted), "oldest record must be evicted once maxBuffered is exceeded")
}

// Reproduces a real production crash: WithLabelValues panics on invalid
// UTF-8, which crashes the entire OM1 process, not just tracing.
func TestStart_invalidUTF8DoesNotPanic(t *testing.T) {
	records := make(chan tracetype.TraceRecord, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	Start(ctx, records, zap.NewNop())

	invalid := "Привіт, я пам'ятаю наші спога\xd1"

	records <- tracetype.TraceRecord{
		Timestamp:  "2026-08-31T21:05:46.177863819Z",
		Generation: 1,
		LLMInput:   invalid,
		LLMOutput:  []map[string]any{},
	}

	waitFor(t, 2*time.Second, func() bool {
		g := metrics.TraceInfo.WithLabelValues("0", "2026-08-31T21:05:46.177863819Z", "1",
			"Привіт, я пам'ятаю наші спога�", "[]")
		return testutil.ToFloat64(g) == 1
	})
}
