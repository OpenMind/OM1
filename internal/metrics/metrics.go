package metrics

import (
	"context"
	"errors"
	"net"
	"net/http"
	"strconv"
	"syscall"

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
		TTSLatency, TTSLatencyLast,
		HTTPProxyParse, HTTPProxyParseLast,
		HTTPUpstreamTotal, HTTPUpstreamTotalLast,
		HTTPUpstreamTTFB, HTTPUpstreamTTFBLast,
		HTTPProxyTotal, HTTPProxyTotalLast,
	)
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
