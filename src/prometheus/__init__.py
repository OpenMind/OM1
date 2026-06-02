import atexit
import logging

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
