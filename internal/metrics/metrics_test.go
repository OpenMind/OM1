package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

func TestRecordHTTPTiming(t *testing.T) {
	RecordHTTPTiming("api.example.com", "/v1/chat", "POST", 200, "12.5", "300", "50", "360")

	gauge := HTTPProxyParseLast.WithLabelValues("api.example.com", "/v1/chat", "POST", "200")
	require.InDelta(t, 0.0125, testutil.ToFloat64(gauge), 1e-9, "milliseconds are converted to seconds")

	total := HTTPUpstreamTotalLast.WithLabelValues("api.example.com", "/v1/chat", "POST", "200")
	require.InDelta(t, 0.300, testutil.ToFloat64(total), 1e-9)
}

func TestRecordHTTPTimingSkipsUnparseable(t *testing.T) {
	gauge := HTTPProxyTotalLast.WithLabelValues("h", "/p", "GET", "404")
	before := testutil.ToFloat64(gauge)
	require.NotPanics(t, func() {
		RecordHTTPTiming("h", "/p", "GET", 404, "not-a-number", "x", "y", "z")
	})
	require.Equal(t, before, testutil.ToFloat64(gauge), "unparseable values are ignored")
}

func TestRecordKBQuerySuccess(t *testing.T) {
	before := testutil.ToFloat64(KBQueries.WithLabelValues("success"))
	RecordKBQuery(0.01, 0.05, true)
	require.Equal(t, before+1, testutil.ToFloat64(KBQueries.WithLabelValues("success")))
	require.InDelta(t, 0.05, testutil.ToFloat64(KBQueryLatencyLast), 1e-9)
	require.InDelta(t, 0.01, testutil.ToFloat64(KBEmbedLatencyLast), 1e-9)
}

func TestRecordKBQueryError(t *testing.T) {
	before := testutil.ToFloat64(KBQueries.WithLabelValues("error"))
	RecordKBQuery(-1, 0.02, false)
	require.Equal(t, before+1, testutil.ToFloat64(KBQueries.WithLabelValues("error")),
		"a failed query increments the error counter")
}

func TestInitQualityLabels(t *testing.T) {
	InitQualityLabels()

	require.Equal(t, 0.0, testutil.ToFloat64(QualityLiveInputClassificationCount.WithLabelValues("negative")),
		"pre-registered at zero -- gives increase() a real prior sample before the label's first real occurrence")
	require.Equal(t, 0.0, testutil.ToFloat64(QualityLiveCoherenceCount.WithLabelValues("marginal")))

	QualityLiveCoherenceCount.WithLabelValues("marginal").Inc()
	InitQualityLabels()
	require.Equal(t, 1.0, testutil.ToFloat64(QualityLiveCoherenceCount.WithLabelValues("marginal")),
		"calling InitQualityLabels again must not reset an already-incremented label")
}

func TestRecordQualityTurn(t *testing.T) {
	beforeLang := testutil.ToFloat64(QualityLiveLanguageCount.WithLabelValues("English"))
	beforeLabel := testutil.ToFloat64(QualityLiveInputClassificationCount.WithLabelValues("positive"))
	beforeCoherence := testutil.ToFloat64(QualityLiveCoherenceCount.WithLabelValues("coherent"))
	beforeTurns := testutil.ToFloat64(QualityLiveTurnsScored)

	RecordQualityTurn("English", "positive", "coherent")

	require.Equal(t, beforeLang+1, testutil.ToFloat64(QualityLiveLanguageCount.WithLabelValues("English")))
	require.Equal(t, beforeLabel+1, testutil.ToFloat64(QualityLiveInputClassificationCount.WithLabelValues("positive")))
	require.Equal(t, beforeCoherence+1, testutil.ToFloat64(QualityLiveCoherenceCount.WithLabelValues("coherent")))
	require.Equal(t, beforeTurns+1, testutil.ToFloat64(QualityLiveTurnsScored),
		"a coherence label was present, so turns_scored increments")
	require.InDelta(t, 1.0, testutil.ToFloat64(QualityLiveActiveScore), 1e-9, "coherent maps to score 1.0")
}

func TestRecordQualityTurnNoResponse(t *testing.T) {
	beforeTurns := testutil.ToFloat64(QualityLiveTurnsScored)
	RecordQualityTurn("English", "not_addressed", "")
	require.Equal(t, beforeTurns, testutil.ToFloat64(QualityLiveTurnsScored),
		"no coherence label (no robot response that turn), so turns_scored does not increment")
}

func TestTraceInfoIsolatedFromDefaultRegistry(t *testing.T) {
	TraceInfo.WithLabelValues("0", "2026-01-01T00:00:00Z", "1", "some prompt text", "[]").Set(1)

	families, err := prometheus.DefaultGatherer.Gather()
	require.NoError(t, err)
	for _, f := range families {
		require.NotEqual(t, "om1_trace_info", f.GetName(),
			"om1_trace_info carries full trace text and must never reach the default /metrics registry that the fleet's Prometheus scrapes")
	}

	families, err = traceRegistry.Gather()
	require.NoError(t, err)
	found := false
	for _, f := range families {
		if f.GetName() == "om1_trace_info" {
			found = true
		}
	}
	require.True(t, found, "om1_trace_info should be registered on traceRegistry, served at /traces/metrics")
}
