package metrics

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

// httpTimingLabels are the labels shared by every om1_http_* timing metric.
var httpTimingLabels = []string{"host", "path", "method", "status_code"}

// metricsAddr is the address the Prometheus metrics server listens on.
const metricsAddr = ":9090"

// LLM metrics.
var (
	LLMLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_llm_latency_seconds",
		Help: "Latency of LLM responses in seconds",
	}, []string{"model", "endpoint"})

	LLMLatencyLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_llm_latency_last_seconds",
		Help: "Most recent LLM response latency in seconds",
	}, []string{"model", "endpoint"})
)

// ASR metrics.
var (
	ASRLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_asr_latency_seconds",
		Help: "Latency from speech activity start to final transcript in seconds",
	}, []string{"model", "language", "api_version"})

	ASRSpeechDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_asr_speech_duration_seconds",
		Help: "Duration of speech activity from speech_start to speech_end in seconds",
	}, []string{"model", "language", "api_version"})

	ASRUtteranceEndLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_asr_utterance_end_latency_seconds",
		Help: "Latency from speech activity start to end_of_utterance detection in seconds",
	}, []string{"model", "language", "api_version"})

	ASRLatencyLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_asr_latency_last_seconds",
		Help: "Most recent latency from speech activity start to final transcript in seconds",
	}, []string{"model", "language", "api_version"})

	ASRSpeechDurationLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_asr_speech_duration_last_seconds",
		Help: "Most recent duration of speech activity from speech_start to speech_end in seconds",
	}, []string{"model", "language", "api_version"})

	ASRUtteranceEndLatencyLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_asr_utterance_end_latency_last_seconds",
		Help: "Most recent latency from speech activity start to end_of_utterance detection in seconds",
	}, []string{"model", "language", "api_version"})

	ASRParallelTranscripts = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "om1_asr_parallel_transcripts_total",
		Help: "Transcripts seen by the parallel ASR sensor, labeled by provider model and first-wins outcome",
	}, []string{"model", "outcome"})
)

// TTS metrics.
var (
	TTSLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_tts_latency_seconds",
		Help: "Latency from TTS synthesis request to first audio chunk in seconds",
	}, []string{"model", "endpoint"})

	TTSLatencyLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_tts_latency_last_seconds",
		Help: "Most recent TTS time-to-first-audio latency in seconds",
	}, []string{"model", "endpoint"})
)

// Knowledge base metrics.
var (
	KBQueryLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name: "om1_kb_query_latency_seconds",
		Help: "Latency of a knowledge base query (embedding plus search) in seconds",
	})
	KBQueryLatencyLast = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "om1_kb_query_latency_last_seconds",
		Help: "Most recent knowledge base query latency in seconds",
	})

	KBEmbedLatency = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name: "om1_kb_embed_latency_seconds",
		Help: "Latency of the embedding step of a knowledge base query in seconds",
	})
	KBEmbedLatencyLast = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "om1_kb_embed_latency_last_seconds",
		Help: "Most recent knowledge base embedding-step latency in seconds",
	})

	KBQueries = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "om1_kb_queries_total",
		Help: "Total number of knowledge base queries by outcome",
	}, []string{"status"})
)

// Startup metrics.
//
// CPU, resident memory and heap-allocation-rate metrics are intentionally NOT
// (re)defined here: client_golang's default registry already exports them via
// the Go runtime and process collectors, so the perf benchmark reads them
// directly off /metrics:
//   - CPU:               process_cpu_seconds_total
//   - Peak RSS:          process_resident_memory_bytes
//   - Heap alloc rate:   rate(go_memstats_alloc_bytes_total[...])
//   - GC / goroutines:   go_gc_duration_seconds, go_goroutines
var (
	StartupDuration = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "om1_startup_duration_seconds",
		Help: "Seconds from process start to the runtime reaching its ready state (cold/warm start time)",
	})
)

// HTTP metrics.
var (
	HTTPProxyParse = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_http_proxy_parse_seconds",
		Help: "Time the proxy spent parsing the request in seconds",
	}, httpTimingLabels)
	HTTPProxyParseLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_http_proxy_parse_last_seconds",
		Help: "Most recent proxy request-parse time in seconds",
	}, httpTimingLabels)

	HTTPUpstreamTotal = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_http_upstream_total_seconds",
		Help: "Total upstream service time in seconds",
	}, httpTimingLabels)
	HTTPUpstreamTotalLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_http_upstream_total_last_seconds",
		Help: "Most recent total upstream service time in seconds",
	}, httpTimingLabels)

	HTTPUpstreamTTFB = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_http_upstream_ttfb_seconds",
		Help: "Upstream time-to-first-byte in seconds",
	}, httpTimingLabels)
	HTTPUpstreamTTFBLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_http_upstream_ttfb_last_seconds",
		Help: "Most recent upstream time-to-first-byte in seconds",
	}, httpTimingLabels)

	HTTPProxyTotal = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_http_proxy_total_seconds",
		Help: "Total proxy time in seconds",
	}, httpTimingLabels)
	HTTPProxyTotalLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_http_proxy_total_last_seconds",
		Help: "Most recent total proxy time in seconds",
	}, httpTimingLabels)
)

func init() {
	prometheus.MustRegister(
		LLMLatency, LLMLatencyLast,
		ASRLatency, ASRLatencyLast,
		ASRSpeechDuration, ASRSpeechDurationLast,
		ASRUtteranceEndLatency, ASRUtteranceEndLatencyLast,
		ASRParallelTranscripts,
		TTSLatency, TTSLatencyLast,
		KBQueryLatency, KBQueryLatencyLast,
		KBEmbedLatency, KBEmbedLatencyLast,
		KBQueries,
		HTTPProxyParse, HTTPProxyParseLast,
		HTTPUpstreamTotal, HTTPUpstreamTotalLast,
		HTTPUpstreamTTFB, HTTPUpstreamTTFBLast,
		HTTPProxyTotal, HTTPProxyTotalLast,
		StartupDuration,
	)
}

// RecordStartupComplete records the time elapsed from process start to the
// runtime reaching its ready state, exposing it as om1_startup_duration_seconds.
// Call this once, when the runtime is initialized and about to enter its main loop.
func RecordStartupComplete(processStart time.Time) {
	StartupDuration.Set(time.Since(processStart).Seconds())
}

// RecordHTTPTiming records HTTP timing metrics for a request with the given attributes and timing values in milliseconds.
// It updates both the histogram and "last" gauge metrics for each timing aspect.
func RecordHTTPTiming(host, path, method string, statusCode int, proxyParseMs, upstreamTotalMs, upstreamTTFBMs, proxyTotalMs string) {
	status := strconv.Itoa(statusCode)
	observe := func(hist *prometheus.HistogramVec, gauge *prometheus.GaugeVec, ms string) {
		v, err := strconv.ParseFloat(ms, 64)
		if err != nil {
			return
		}
		seconds := v / 1000.0
		hist.WithLabelValues(host, path, method, status).Observe(seconds)
		gauge.WithLabelValues(host, path, method, status).Set(seconds)
	}
	observe(HTTPProxyParse, HTTPProxyParseLast, proxyParseMs)
	observe(HTTPUpstreamTotal, HTTPUpstreamTotalLast, upstreamTotalMs)
	observe(HTTPUpstreamTTFB, HTTPUpstreamTTFBLast, upstreamTTFBMs)
	observe(HTTPProxyTotal, HTTPProxyTotalLast, proxyTotalMs)
}

// RecordKBQuery records the latency of a knowledge base query.
func RecordKBQuery(embedSeconds, querySeconds float64, ok bool) {
	status := "success"
	if !ok {
		status = "error"
	}
	KBQueries.WithLabelValues(status).Inc()

	KBQueryLatency.Observe(querySeconds)
	KBQueryLatencyLast.Set(querySeconds)

	if embedSeconds >= 0 {
		KBEmbedLatency.Observe(embedSeconds)
		KBEmbedLatencyLast.Set(embedSeconds)
	}
}

// StartServer starts the HTTP server that exposes the Prometheus metrics on port 9090.
func StartServer(log *zap.Logger) func(context.Context) error {
	noop := func(context.Context) error { return nil }

	ln, err := net.Listen("tcp", metricsAddr)
	if err != nil {
		if errors.Is(err, syscall.EADDRINUSE) {
			log.Warn("Prometheus port 9090 already in use, reusing existing server")
			return noop
		}
		log.Warn("failed to start Prometheus metrics server", zap.Error(err))
		return noop
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	srv := &http.Server{Handler: mux}

	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Warn("Prometheus metrics server error", zap.Error(err))
		}
	}()
	log.Info("Prometheus metrics server started on port 9090")

	return func(ctx context.Context) error {
		err := srv.Shutdown(ctx)
		log.Info("Prometheus metrics server stopped")
		return err
	}
}
