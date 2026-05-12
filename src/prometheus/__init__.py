from prometheus_client import Histogram, start_http_server

start_http_server(9191)

om1_llm_latency = Histogram(
    "om1_llm_latency_seconds",
    "Latency of LLM responses in seconds",
    ["model", "endpoint"],
)
