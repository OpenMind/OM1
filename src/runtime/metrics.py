"""Prometheus metrics for OM1 runtime monitoring.

Provides metric definitions for inputs, actions, backgrounds, simulators,
cortex ticks, and mode transitions. Falls back to no-op stubs when
prometheus_client is not installed.

Start the metrics HTTP server with ``start_metrics_server()``.
The server port is controlled by the ``METRICS_PORT`` environment variable
(default 9464).

Endpoints:
- ``/`` — Human-readable HTML dashboard (auto-refreshes every 5s)
- ``/metrics`` — Prometheus scrape endpoint (raw text format)
"""

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Union

# ---------------------------------------------------------------------------
# Metric category definitions for the HTML dashboard
# ---------------------------------------------------------------------------

_CATEGORIES: dict[str, dict[str, str]] = {
    "Inputs": {
        "om1_input_status": "Status",
        "om1_input_events_total": "Events",
        "om1_input_errors_total": "Errors",
    },
    "Actions": {
        "om1_action_status": "Status",
        "om1_action_executions_total": "Executions",
        "om1_action_errors_total": "Errors",
    },
    "Backgrounds": {
        "om1_background_status": "Status",
        "om1_background_runs_total": "Runs",
        "om1_background_errors_total": "Errors",
    },
    "Simulators": {
        "om1_simulator_status": "Status",
        "om1_simulator_ticks_total": "Ticks",
        "om1_simulator_errors_total": "Errors",
    },
    "Cortex": {
        "om1_cortex_ticks_total": "Ticks",
        "om1_cortex_tick_duration_seconds": "Tick Duration (s)",
    },
    "Mode": {
        "om1_mode_current": "Current Mode",
        "om1_mode_transitions_total": "Transitions",
    },
}

# ---------------------------------------------------------------------------
# HTML dashboard renderer
# ---------------------------------------------------------------------------

_STATUS_METRICS = {
    "om1_input_status",
    "om1_action_status",
    "om1_background_status",
    "om1_simulator_status",
    "om1_mode_current",
}

# Suffixes to skip when collecting samples (histogram buckets, counter created)
_SKIP_SUFFIXES = ("_bucket", "_created")


def _render_dashboard() -> str:
    """Collect metrics from the registry and return an HTML dashboard string."""
    from prometheus_client import REGISTRY

    # Collect all om1_* samples keyed by sample name.
    # Each entry: [(labels_dict, value), ...]
    samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if not sample.name.startswith("om1_"):
                continue
            if sample.name.endswith(_SKIP_SUFFIXES):
                continue
            samples.setdefault(sample.name, []).append(
                (dict(sample.labels), sample.value)
            )

    # Build category sections
    sections: list[str] = []
    for cat_title, metric_map in _CATEGORIES.items():
        rows: list[str] = []
        for metric_name, display_name in metric_map.items():
            # Resolve matching sample names.
            # Gauges/Counters: direct match.
            # Histograms: base name won't exist; look for _count and _sum.
            matched: list[tuple[str, str]] = []
            if metric_name in samples:
                matched.append((metric_name, display_name))
            else:
                count_key = metric_name + "_count"
                sum_key = metric_name + "_sum"
                if count_key in samples:
                    matched.append((count_key, display_name + " Count"))
                if sum_key in samples:
                    matched.append((sum_key, display_name + " Sum"))

            is_status = metric_name in _STATUS_METRICS
            for sample_name, col_label in matched:
                for labels, value in samples[sample_name]:
                    label_parts = [v for k, v in sorted(labels.items()) if k != "le"]
                    label_str = ", ".join(label_parts) if label_parts else "-"
                    if is_status:
                        if value == 1:
                            val_cell = '<span class="dot green"></span> running'
                        elif value == -1:
                            val_cell = '<span class="dot red"></span> failed'
                        else:
                            val_cell = '<span class="dot gray"></span> stopped'
                    else:
                        val_cell = (
                            f"{value:.4f}" if value != int(value) else str(int(value))
                        )
                    rows.append(
                        f"<tr><td>{col_label}</td>"
                        f"<td>{label_str}</td>"
                        f"<td>{val_cell}</td></tr>"
                    )
        if rows:
            sections.append(
                f"<h2>{cat_title}</h2>"
                f"<table><tr><th>Metric</th><th>Labels</th><th>Value</th></tr>"
                f"{''.join(rows)}</table>"
            )

    body = "".join(sections) if sections else "<p>No om1 metrics recorded yet.</p>"

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta http-equiv='refresh' content='5'>"
        "<title>OM1 Metrics</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;background:#f8f9fa}"
        "h1{color:#1a1a2e}"
        "h2{color:#16213e;margin-top:1.5rem}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1rem}"
        "th,td{text-align:left;padding:.4rem .8rem;border:1px solid #dee2e6}"
        "th{background:#e9ecef}"
        "tr:nth-child(even){background:#f1f3f5}"
        ".dot{display:inline-block;width:10px;height:10px;border-radius:50%;"
        "margin-right:6px;vertical-align:middle}"
        ".green{background:#2ecc71}.red{background:#e74c3c}.gray{background:#95a5a6}"
        "</style></head><body>"
        "<h1>OM1 Metrics Dashboard</h1>"
        f"{body}"
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# No-op stub used when prometheus_client is not installed
# ---------------------------------------------------------------------------


class _NoOp:
    """Drop-in stub that silently ignores all metric operations."""

    def labels(self, **_kwargs: str) -> "_NoOp":
        return self

    def inc(self, _amount: float = 1) -> None:
        return None

    def set(self, _value: float) -> None:
        return None

    def observe(self, _value: float) -> None:
        return None


# ---------------------------------------------------------------------------
# Metric definitions + HTTP server
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _MetricsHandler(BaseHTTPRequestHandler):
        """Serves HTML dashboard on ``/`` and Prometheus format on ``/metrics``."""

        def do_GET(self) -> None:
            if self.path == "/metrics":
                data = generate_latest()
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/":
                html = _render_dashboard().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress default stderr access logs."""

    # -- Input metrics --
    INPUT_STATUS: Union[Gauge, _NoOp] = Gauge(
        "om1_input_status",
        "Input status (1=running, -1=failed)",
        ["name"],
    )
    INPUT_EVENTS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_input_events_total",
        "Total input events processed",
        ["name"],
    )
    INPUT_ERRORS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_input_errors_total",
        "Total input errors",
        ["name"],
    )

    # -- Action metrics --
    ACTION_STATUS: Union[Gauge, _NoOp] = Gauge(
        "om1_action_status",
        "Action connector status (1=running, 0=stopped)",
        ["name"],
    )
    ACTION_EXECUTIONS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_action_executions_total",
        "Total action connect() calls",
        ["name"],
    )
    ACTION_ERRORS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_action_errors_total",
        "Total action errors",
        ["name"],
    )

    # -- Background metrics --
    BACKGROUND_STATUS: Union[Gauge, _NoOp] = Gauge(
        "om1_background_status",
        "Background task status (1=running, 0=stopped)",
        ["name"],
    )
    BACKGROUND_RUNS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_background_runs_total",
        "Total background run() calls",
        ["name"],
    )
    BACKGROUND_ERRORS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_background_errors_total",
        "Total background errors",
        ["name"],
    )

    # -- Simulator metrics --
    SIMULATOR_STATUS: Union[Gauge, _NoOp] = Gauge(
        "om1_simulator_status",
        "Simulator status (1=running, 0=stopped)",
        ["name"],
    )
    SIMULATOR_TICKS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_simulator_ticks_total",
        "Total simulator tick() calls",
        ["name"],
    )
    SIMULATOR_ERRORS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_simulator_errors_total",
        "Total simulator errors",
        ["name"],
    )

    # -- Cortex metrics --
    CORTEX_TICKS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_cortex_ticks_total",
        "Total cortex tick count",
    )
    CORTEX_TICK_DURATION: Union[Histogram, _NoOp] = Histogram(
        "om1_cortex_tick_duration_seconds",
        "Cortex tick duration in seconds",
    )

    # -- Mode metrics --
    MODE_CURRENT: Union[Gauge, _NoOp] = Gauge(
        "om1_mode_current",
        "Currently active mode (1=active)",
        ["mode"],
    )
    MODE_TRANSITIONS_TOTAL: Union[Counter, _NoOp] = Counter(
        "om1_mode_transitions_total",
        "Total mode transitions",
        ["from_mode", "to_mode"],
    )

    def start_metrics_server() -> None:
        """Start the metrics HTTP server with dashboard and Prometheus endpoints."""
        port = int(os.environ.get("METRICS_PORT", "9464"))
        try:
            server = HTTPServer(("", port), _MetricsHandler)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            logging.info(
                f"Metrics server started on port {port} "
                f"(dashboard: http://localhost:{port}/)"
            )
        except OSError as e:
            logging.warning(f"Failed to start metrics server on port {port}: {e}")

except ImportError:
    _noop = _NoOp()

    INPUT_STATUS = _noop
    INPUT_EVENTS_TOTAL = _noop
    INPUT_ERRORS_TOTAL = _noop

    ACTION_STATUS = _noop
    ACTION_EXECUTIONS_TOTAL = _noop
    ACTION_ERRORS_TOTAL = _noop

    BACKGROUND_STATUS = _noop
    BACKGROUND_RUNS_TOTAL = _noop
    BACKGROUND_ERRORS_TOTAL = _noop

    SIMULATOR_STATUS = _noop
    SIMULATOR_TICKS_TOTAL = _noop
    SIMULATOR_ERRORS_TOTAL = _noop

    CORTEX_TICKS_TOTAL = _noop
    CORTEX_TICK_DURATION = _noop

    MODE_CURRENT = _noop
    MODE_TRANSITIONS_TOTAL = _noop

    def start_metrics_server() -> None:
        """No-op when prometheus_client is not installed."""
        logging.info("prometheus_client not installed, metrics server disabled")
