import atexit
import logging
import os
import sys
import time

from prometheus_client import Gauge, Histogram, start_http_server

_prometheus_server = None
_prometheus_thread = None


def _stop_prometheus_server():
    """
    Stop the Prometheus server on program exit.
    """
    global _prometheus_server, _prometheus_thread
    if _prometheus_server is not None:
        try:
            _prometheus_server.shutdown()
            if _prometheus_thread is not None:
                _prometheus_thread.join(timeout=2.0)
            logging.info("Prometheus metrics server stopped")
        except Exception as e:
            logging.warning(f"Error stopping Prometheus server: {e}")


try:
    result = start_http_server(9090)
    _prometheus_server, _prometheus_thread = result

    atexit.register(_stop_prometheus_server)
    logging.info("Prometheus metrics server started on port 9090")
except OSError as e:
    if "Address already in use" in str(e):
        logging.warning("Prometheus port 9090 already in use, reusing existing server")
    else:
        raise

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

# Resource Metrics
#
# CPU and resident memory are already exported by prometheus_client's default
# ProcessCollector (process_cpu_seconds_total, process_resident_memory_bytes),
# so the perf benchmark reads those directly off /metrics. The two metrics below
# add the pieces the default collectors do not provide.

# Cold/warm start time: seconds from interpreter start to the runtime being ready.
om1_startup_duration = Gauge(
    "om1_startup_duration_seconds",
    "Seconds from process start to the runtime reaching its ready state (cold/warm start time)",
)

# Python heap-allocation pressure. CPython has no direct "alloc rate" counter, so
# we expose the number of currently allocated memory blocks; rate() of this in
# Prometheus is the closest analog to the Go branch's go_memstats_alloc_bytes_total.
# Evaluated at scrape time via set_function, so it costs nothing between scrapes.
om1_python_allocated_blocks = Gauge(
    "om1_python_allocated_blocks",
    "Number of currently allocated CPython memory blocks (alloc-pressure proxy)",
)
om1_python_allocated_blocks.set_function(sys.getallocatedblocks)

# Optional byte-accurate heap tracking via tracemalloc, gated behind an env var
# because it adds measurable per-allocation overhead. Enable with OM1_TRACEMALLOC=1
# during a benchmark run to get om1_python_traced_heap_bytes.
if os.environ.get("OM1_TRACEMALLOC") == "1":
    import tracemalloc

    if not tracemalloc.is_tracing():
        tracemalloc.start()
    om1_python_traced_heap_bytes = Gauge(
        "om1_python_traced_heap_bytes",
        "Current size of Python objects tracked by tracemalloc, in bytes",
    )
    om1_python_traced_heap_bytes.set_function(lambda: tracemalloc.get_traced_memory()[0])

# Interpreter start time, used by record_startup_complete to derive startup duration.
_process_start = time.time()


def record_startup_complete():
    """Record the cold/warm start time as om1_startup_duration_seconds.

    Call once, when the runtime is initialized and about to enter its main loop.
    """
    om1_startup_duration.set(time.time() - _process_start)


# Per-request roll-up metrics, labeled by kind (llm, asr, tts).
#
# om1_request_total_seconds is the full client-side time for one outbound request,
# measured from the start of building the request to the end of parsing the
# response (build + travel + proxy + vendor + parse). om1_request_proxy_seconds is
# the gateway time for that request, from the x-proxy-total-ms response header
# (recorded in the shared httpx hook). So
# (om1_request_total_seconds - om1_request_proxy_seconds) ≈ OM1 compute + travel.
om1_request_total = Histogram(
    "om1_request_total_seconds",
    "Total client-side request time (build+travel+proxy+parse) in seconds",
    ["kind"],
)
om1_request_total_last = Gauge(
    "om1_request_total_last_seconds",
    "Most recent total client-side request time in seconds",
    ["kind"],
)

om1_request_proxy = Histogram(
    "om1_request_proxy_seconds",
    "Gateway proxy time for a request (from x-proxy-total-ms) in seconds",
    ["kind"],
)
om1_request_proxy_last = Gauge(
    "om1_request_proxy_last_seconds",
    "Most recent gateway proxy time for a request in seconds",
    ["kind"],
)


def record_request_total(kind, seconds):
    """Record the full client-side time for one outbound request (kind=llm/asr/tts)."""
    om1_request_total.labels(kind=kind).observe(seconds)
    om1_request_total_last.labels(kind=kind).set(seconds)


def record_request_proxy(kind, seconds):
    """Record the gateway proxy time for one outbound request (kind=llm/asr/tts)."""
    om1_request_proxy.labels(kind=kind).observe(seconds)
    om1_request_proxy_last.labels(kind=kind).set(seconds)
