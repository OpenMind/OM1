"""Prometheus metrics for OM1 runtime monitoring.

Provides metric definitions for inputs, actions, backgrounds, simulators,
cortex ticks, and mode transitions. Falls back to no-op stubs when
prometheus_client is not installed.

Start the metrics HTTP server with ``start_metrics_server()``.
The server port is controlled by the ``METRICS_PORT`` environment variable
(default 9090).
"""

import logging
import os
from typing import Union

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
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

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
        """Start the Prometheus metrics HTTP server."""
        port = int(os.environ.get("METRICS_PORT", "9090"))
        try:
            start_http_server(port)
            logging.info(f"Metrics server started on port {port}")
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
