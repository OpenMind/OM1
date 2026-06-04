package metrics

import (
	"testing"

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
