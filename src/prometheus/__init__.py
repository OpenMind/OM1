from prometheus_client import Gauge, Histogram, start_http_server

start_http_server(9090)

# LLM Metrics
om1_llm_latency = Histogram(
    "om1_llm_latency_seconds",
    "Latency of LLM responses in seconds",
    ["model", "endpoint"],
)

om1_llm_latency_last = Gauge(
    "om1_llm_latency_last_seconds",
    "Most recent LLM response latency in seconds",
    ["model", "endpoint"],
)

# ASR Metrics
om1_asr_latency = Histogram(
    "om1_asr_latency_seconds",
    "Latency from speech activity start to final transcript in seconds",
    ["model", "language", "api_version"],
)

om1_asr_speech_duration = Histogram(
    "om1_asr_speech_duration_seconds",
    "Duration of speech activity from speech_start to speech_end in seconds",
    ["model", "language", "api_version"],
)

om1_asr_utterance_end_latency = Histogram(
    "om1_asr_utterance_end_latency_seconds",
    "Latency from speech activity start to end_of_utterance detection in seconds",
    ["model", "language", "api_version"],
)

om1_asr_latency_last = Gauge(
    "om1_asr_latency_last_seconds",
    "Most recent latency from speech activity start to final transcript in seconds",
    ["model", "language", "api_version"],
)

om1_asr_speech_duration_last = Gauge(
    "om1_asr_speech_duration_last_seconds",
    "Most recent duration of speech activity from speech_start to speech_end in seconds",
    ["model", "language", "api_version"],
)

om1_asr_utterance_end_latency_last = Gauge(
    "om1_asr_utterance_end_latency_last_seconds",
    "Most recent latency from speech activity start to end_of_utterance detection in seconds",
    ["model", "language", "api_version"],
)

om1_http_request_duration_seconds = Histogram(
    "om1_http_request_duration_seconds",
    "Total HTTP request duration (client-side) in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_total_seconds = Histogram(
    "om1_http_upstream_total_seconds",
    "Upstream total time in seconds (from x-upstream-total-ms header)",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_ttfb_seconds = Histogram(
    "om1_http_upstream_ttfb_seconds",
    "Upstream TTFB in seconds (from x-upstream-ttfb-ms header)",
    ["host", "path", "method", "status_code"],
)

om1_http_proxy_total_seconds = Histogram(
    "om1_http_proxy_total_seconds",
    "Proxy total time in seconds (from x-proxy-total-ms header)",
    ["host", "path", "method", "status_code"],
)

om1_http_request_duration_last_seconds = Gauge(
    "om1_http_request_duration_last_seconds",
    "Most recent HTTP request duration (client-side) in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_total_last_seconds = Gauge(
    "om1_http_upstream_total_last_seconds",
    "Most recent upstream total time in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_upstream_ttfb_last_seconds = Gauge(
    "om1_http_upstream_ttfb_last_seconds",
    "Most recent upstream TTFB in seconds",
    ["host", "path", "method", "status_code"],
)

om1_http_proxy_total_last_seconds = Gauge(
    "om1_http_proxy_total_last_seconds",
    "Most recent proxy total time in seconds",
    ["host", "path", "method", "status_code"],
)
