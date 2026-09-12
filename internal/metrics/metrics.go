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

	"github.com/openmind/om1/internal/logger"
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

// VLM metrics.
var (
	VLMLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_vlm_latency_seconds",
		Help: "Latency of VLM (vision language model) responses in seconds",
	}, []string{"model", "endpoint"})

	VLMLatencyLast = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "om1_vlm_latency_last_seconds",
		Help: "Most recent VLM response latency in seconds",
	}, []string{"model", "endpoint"})
)

// ASR metrics.
var (
	ASRLatency = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "om1_asr_latency_seconds",
		Help: "Latency from locally-detected (VAD) end-of-speech to final transcript in seconds.",
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
		Help: "Most recent latency from locally-detected (VAD) end-of-speech to final transcript in seconds.",
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

// Quality scorer metrics
var (
	QualityLiveLanguageCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "om1_quality_live_language_count",
		Help: "Per-run count of scored turns by detected spoken language",
	}, []string{"language"})

	QualityLiveInputClassificationCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "om1_quality_live_input_classification_count",
		Help: "Per-run count of user inputs by classification label (positive/marginal/negative/not_addressed)",
	}, []string{"label"})

	QualityLiveCoherenceCount = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "om1_quality_live_coherence_count",
		Help: "Per-run count of prompt/response pairs by coherence label (coherent/marginal/incoherent)",
	}, []string{"coherence"})

	QualityLiveTurnsScored = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "om1_quality_live_turns_scored",
		Help: "Per-run count of turns that produced a coherence score",
	})

	QualityLiveActiveScore = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "om1_quality_live_active_score",
		Help: "Most recent turn's coherence score: coherent=1, marginal=0.5, incoherent=0",
	})
)

// qualityCoherenceScore maps a coherence label to the om1_quality_live_active_score value, mirroring COHERENCE_SCORE.
var qualityCoherenceScore = map[string]float64{
	"incoherent": 0.0,
	"marginal":   0.5,
	"coherent":   1.0,
}

// TraceInfo is an "info" metric (value always 1, all content in labels) --
// the same pattern as kube_pod_info.
var TraceInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
	Name: "om1_trace_info",
	Help: "One series per buffered LLM trace record awaiting export by the telemetry sidecar; value is always 1, full record in labels.",
}, []string{"seq", "ts", "generation", "llm_input", "llm_output"})

func init() {
	prometheus.MustRegister(
		TraceInfo,
		LLMLatency, LLMLatencyLast,
		VLMLatency, VLMLatencyLast,
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
		QualityLiveLanguageCount,
		QualityLiveInputClassificationCount,
		QualityLiveCoherenceCount,
		QualityLiveTurnsScored,
		QualityLiveActiveScore,
	)
}

// InitQualityLabels pre-registers known classification/coherence labels at zero, so their first real increment isn't invisible to increase() over short windows.
func InitQualityLabels() {
	for _, label := range []string{"positive", "marginal", "negative", "not_addressed"} {
		QualityLiveInputClassificationCount.WithLabelValues(label).Add(0)
	}
	for _, coherence := range []string{"coherent", "marginal", "incoherent"} {
		QualityLiveCoherenceCount.WithLabelValues(coherence).Add(0)
	}
}

// RecordQualityTurn records the live quality-scorer's classification of one conversation turn.
func RecordQualityTurn(language, classification, coherence string) {
	if language != "" {
		QualityLiveLanguageCount.WithLabelValues(language).Inc()
	}
	if classification != "" {
		QualityLiveInputClassificationCount.WithLabelValues(classification).Inc()
	}
	if coherence != "" {
		QualityLiveCoherenceCount.WithLabelValues(coherence).Inc()
		QualityLiveTurnsScored.Inc()
		QualityLiveActiveScore.Set(qualityCoherenceScore[coherence])
	}
}

// RecordHTTPTiming records HTTP timing metrics for a request with the given attributes and timing values in milliseconds.
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

// RecordLLMResponseLatency records the latency of an LLM response along with relevant request and response attributes.
func RecordResponseLatency(latency *prometheus.HistogramVec, latencyLast *prometheus.GaugeVec, provider, model, endpoint string, req *http.Request, resp *http.Response, start time.Time) {
	header := func(key string) string {
		if v := resp.Header.Get(key); v != "" {
			return v
		}
		return "?"
	}
	elapsed := time.Since(start)
	logger.Get().Info(provider,
		zap.String("method", req.Method),
		zap.String("url", req.URL.String()),
		zap.Int("status", resp.StatusCode),
		zap.Int64("elapsed_ms", elapsed.Milliseconds()),
		zap.String("proxy_parse_ms", header("x-proxy-parse-ms")),
		zap.String("upstream_total_ms", header("x-upstream-total-ms")),
		zap.String("upstream_ttfb_ms", header("x-upstream-ttfb-ms")),
		zap.String("proxy_total_ms", header("x-proxy-total-ms")),
	)

	seconds := elapsed.Seconds()
	latency.WithLabelValues(model, endpoint).Observe(seconds)
	latencyLast.WithLabelValues(model, endpoint).Set(seconds)

	RecordHTTPTiming(req.URL.Host, req.URL.Path, req.Method, resp.StatusCode,
		header("x-proxy-parse-ms"), header("x-upstream-total-ms"),
		header("x-upstream-ttfb-ms"), header("x-proxy-total-ms"))
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
