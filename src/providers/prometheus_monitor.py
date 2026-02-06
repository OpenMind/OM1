"""Prometheus-based health monitoring for OM1 providers."""

import html
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, cast

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from .singleton import singleton

# Module-level metrics (created once, reused across singleton resets)
_metrics_initialized = False
_status_gauge: Optional[Gauge] = None
_heartbeat_gauge: Optional[Gauge] = None
_error_counter: Optional[Counter] = None
_providers_total: Optional[Gauge] = None
_uptime_gauge: Optional[Gauge] = None


def _init_metrics() -> None:
    """Initialize Prometheus metrics (only once per process)."""
    global _metrics_initialized, _status_gauge, _heartbeat_gauge
    global _error_counter, _providers_total, _uptime_gauge

    if _metrics_initialized:
        return

    _status_gauge = Gauge(
        "om1_provider_status",
        "Provider health status (1=active, 0=inactive)",
        ["provider", "type", "category", "status"],
    )
    _heartbeat_gauge = Gauge(
        "om1_provider_seconds_since_heartbeat",
        "Seconds since last heartbeat from provider",
        ["provider", "type", "category"],
    )
    _error_counter = Counter(
        "om1_provider_errors_total",
        "Total errors reported by provider",
        ["provider", "type", "category"],
    )
    _providers_total = Gauge(
        "om1_providers_total",
        "Total number of providers by status",
        ["status"],
    )
    _uptime_gauge = Gauge(
        "om1_uptime_seconds",
        "System uptime in seconds",
    )
    _metrics_initialized = True


class HealthStatus(Enum):
    """Health status enum for providers."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderState:
    """Internal state for a registered provider."""

    name: str
    metadata: Dict[str, str] = field(default_factory=dict)
    last_heartbeat: float = 0.0
    error_count: int = 0
    status: HealthStatus = HealthStatus.UNKNOWN


@singleton
class PrometheusMonitor:
    """
    Prometheus-based health monitoring singleton for OM1 providers.

    Provides metrics collection, heartbeat tracking, and error counting
    for registered providers.

    Parameters
    ----------
    heartbeat_timeout : float
        Seconds without heartbeat before marking provider unhealthy.
        Default is 30.0.
    check_interval : float
        Interval in seconds between health checks. Default is 5.0.
    """

    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        check_interval: float = 5.0,
    ):
        """Initialize PrometheusMonitor with metrics and internal state."""
        # Initialize module-level metrics (only once per process)
        _init_metrics()

        self._lock = threading.Lock()
        self._providers: Dict[str, ProviderState] = {}
        self._start_time = time.time()
        self._heartbeat_timeout = heartbeat_timeout
        self._check_interval = check_interval
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._server_started = False

        # Reference module-level metrics (guaranteed non-None after _init_metrics)
        self._status_gauge: Gauge = cast(Gauge, _status_gauge)
        self._heartbeat_gauge: Gauge = cast(Gauge, _heartbeat_gauge)
        self._error_counter: Counter = cast(Counter, _error_counter)
        self._providers_total: Gauge = cast(Gauge, _providers_total)
        self._uptime_gauge: Gauge = cast(Gauge, _uptime_gauge)

        logging.info("PrometheusMonitor initialized")

    def start(self, port: int = 9090) -> None:
        """
        Start the Prometheus HTTP server and health check thread.

        Parameters
        ----------
        port : int
            Port number for the /metrics endpoint. Default is 9090.
        """
        with self._lock:
            if not self._server_started:
                try:
                    self._start_dashboard_server(port)
                    self._server_started = True
                    logging.info(f"Health dashboard started on port {port}")
                except OSError as e:
                    logging.warning(f"Could not start metrics server: {e}")

            if not self._running:
                self._running = True
                self._check_thread = threading.Thread(
                    target=self._health_check_loop, daemon=True
                )
                self._check_thread.start()
                logging.info("Health check thread started")

    def _start_dashboard_server(self, port: int) -> None:
        """Start FastAPI server with health dashboard and metrics endpoint."""
        app = FastAPI(title="OM1 Health Dashboard")

        @app.get("/", response_class=HTMLResponse)
        async def health_dashboard() -> str:
            return self._generate_dashboard_html()

        @app.get("/metrics")
        async def metrics() -> Response:
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @app.get("/api/health")
        async def health_api() -> dict:
            return self._get_health_data()

        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="error",
        )
        server = uvicorn.Server(config)
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

    def _get_health_data(self) -> dict:
        """Get health data for API endpoint."""
        current_time = time.time()
        providers_data = []

        with self._lock:
            for name, provider in self._providers.items():
                seconds_since_heartbeat = current_time - provider.last_heartbeat
                providers_data.append(
                    {
                        "name": name,
                        "type": provider.metadata.get("type", "unknown"),
                        "category": provider.metadata.get("category", "unknown"),
                        "status": provider.status.value,
                        "last_heartbeat": seconds_since_heartbeat,
                        "error_count": provider.error_count,
                    }
                )

        healthy = sum(1 for p in providers_data if p["status"] == "healthy")
        unhealthy = sum(1 for p in providers_data if p["status"] == "unhealthy")
        unknown = sum(1 for p in providers_data if p["status"] == "unknown")

        return {
            "uptime": current_time - self._start_time,
            "healthy_count": healthy,
            "unhealthy_count": unhealthy,
            "unknown_count": unknown,
            "providers": providers_data,
        }

    def _generate_dashboard_html(self) -> str:
        """Generate HTML dashboard for health monitoring."""
        data = self._get_health_data()
        uptime_min = int(data["uptime"] / 60)
        uptime_sec = int(data["uptime"] % 60)

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>OM1 Health</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            background: #0d1117;
            color: #c9d1d9;
            font-size: 13px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            border-bottom: 1px solid #21262d;
            margin-bottom: 24px;
        }}
        .header h1 {{ font-size: 16px; font-weight: 600; color: #f0f6fc; }}
        .header-stats {{ display: flex; gap: 24px; font-size: 12px; }}
        .header-stats span {{ color: #8b949e; }}
        .header-stats .value {{ font-weight: 600; margin-left: 4px; }}
        .header-stats .ok {{ color: #3fb950; }}
        .header-stats .err {{ color: #f85149; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{
            text-align: left;
            padding: 8px 12px;
            color: #8b949e;
            font-weight: 500;
            font-size: 12px;
            border-bottom: 1px solid #21262d;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #21262d;
        }}
        tr:hover {{ background: #161b22; }}
        .status {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status.ok {{ background: #3fb950; }}
        .status.err {{ background: #f85149; }}
        .name {{ color: #f0f6fc; font-weight: 500; }}
        .type {{ color: #8b949e; }}
        .category {{
            display: inline-block;
            padding: 2px 8px;
            background: #21262d;
            border-radius: 12px;
            font-size: 11px;
            color: #8b949e;
        }}
        .num {{ font-variant-numeric: tabular-nums; color: #8b949e; }}
        .num.warn {{ color: #d29922; }}
        .footer {{
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid #21262d;
            font-size: 12px;
            color: #484f58;
        }}
        .footer a {{ color: #58a6ff; text-decoration: none; }}
        .footer a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OM1 Health</h1>
            <div class="header-stats">
                <span>Uptime<span class="value">{uptime_min}m {uptime_sec}s</span></span>
                <span>Healthy<span class="value ok">{data['healthy_count']}</span></span>
                <span>Unhealthy<span class="value err">{data['unhealthy_count']}</span></span>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Status</th>
                    <th>Provider</th>
                    <th>Type</th>
                    <th>Category</th>
                    <th>Heartbeat</th>
                    <th>Errors</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_table_rows(data['providers'])}
            </tbody>
        </table>
        <div class="footer">
            <a href="/metrics">/metrics</a> · <a href="/api/health">/api/health</a>
        </div>
    </div>
</body>
</html>
"""

    def _generate_table_rows(self, providers: list) -> str:
        """Generate table rows for providers."""
        rows = ""
        for p in sorted(providers, key=lambda x: (x["category"], x["name"])):
            status_class = "ok" if p["status"] == "healthy" else "err"
            heartbeat = f"{p['last_heartbeat']:.1f}s"
            error_class = "warn" if p["error_count"] > 0 else ""
            name = html.escape(str(p["name"]))
            ptype = html.escape(str(p["type"]))
            category = html.escape(str(p["category"]))
            rows += f"""
                <tr>
                    <td><span class="status {status_class}"></span></td>
                    <td class="name">{name}</td>
                    <td class="type">{ptype}</td>
                    <td><span class="category">{category}</span></td>
                    <td class="num">{heartbeat}</td>
                    <td class="num {error_class}">{p['error_count']}</td>
                </tr>
            """
        return rows

    def stop(self) -> None:
        """Stop the health check thread."""
        with self._lock:
            self._running = False
            if self._check_thread:
                self._check_thread.join(timeout=5.0)
                self._check_thread = None
        logging.info("PrometheusMonitor stopped")

    def register(
        self,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
        recovery_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Register a provider for health monitoring.

        Parameters
        ----------
        name : str
            Unique name for the provider.
        metadata : dict, optional
            Additional metadata about the provider.
        recovery_callback : callable, optional
            Deprecated. Kept for backward compatibility, ignored.
        """
        with self._lock:
            if name in self._providers:
                logging.debug(f"Provider {name} already registered, updating")

            provider_metadata = metadata or {}
            provider_type = provider_metadata.get("type", "unknown")
            provider_category = provider_metadata.get("category", "unknown")

            self._providers[name] = ProviderState(
                name=name,
                metadata=provider_metadata,
                last_heartbeat=time.time(),
                status=HealthStatus.HEALTHY,
            )

            # Initialize metrics for this provider with type and category labels
            self._status_gauge.labels(
                provider=name,
                type=provider_type,
                category=provider_category,
                status="healthy",
            ).set(1)
            self._status_gauge.labels(
                provider=name,
                type=provider_type,
                category=provider_category,
                status="unhealthy",
            ).set(0)
            self._heartbeat_gauge.labels(
                provider=name, type=provider_type, category=provider_category
            ).set(0)

            logging.info(
                f"Registered provider: {name} (type={provider_type}, category={provider_category})"
            )

    def unregister(self, name: str) -> None:
        """
        Unregister a provider from health monitoring.

        Parameters
        ----------
        name : str
            Name of the provider to unregister.
        """
        with self._lock:
            if name in self._providers:
                provider = self._providers[name]
                provider_type = provider.metadata.get("type", "unknown")
                provider_category = provider.metadata.get("category", "unknown")

                try:
                    self._status_gauge.labels(
                        provider=name,
                        type=provider_type,
                        category=provider_category,
                        status="healthy",
                    ).set(0)
                    self._status_gauge.labels(
                        provider=name,
                        type=provider_type,
                        category=provider_category,
                        status="unhealthy",
                    ).set(0)
                    self._heartbeat_gauge.labels(
                        provider=name,
                        type=provider_type,
                        category=provider_category,
                    ).set(0)
                except Exception as e:
                    logging.debug(f"Error cleaning up metrics for {name}: {e}")

                del self._providers[name]
                logging.info(f"Unregistered provider: {name}")

    def heartbeat(self, name: str) -> None:
        """
        Record a heartbeat from a provider.

        Parameters
        ----------
        name : str
            Name of the provider sending heartbeat.
        """
        with self._lock:
            if name in self._providers:
                provider = self._providers[name]
                provider.last_heartbeat = time.time()
                if provider.status != HealthStatus.HEALTHY:
                    provider.status = HealthStatus.HEALTHY
                    self._update_status_metrics(name, HealthStatus.HEALTHY)
                    logging.info(f"Provider {name} recovered (heartbeat received)")

    def report_error(self, name: str, error: str) -> None:
        """
        Report an error from a provider.

        Parameters
        ----------
        name : str
            Name of the provider reporting error.
        error : str
            Error message description.
        """
        with self._lock:
            if name in self._providers:
                provider = self._providers[name]
                provider.error_count += 1
                provider_type = provider.metadata.get("type", "unknown")
                provider_category = provider.metadata.get("category", "unknown")
                self._error_counter.labels(
                    provider=name, type=provider_type, category=provider_category
                ).inc()
                logging.warning(f"Provider {name} error: {error}")

    def get_status(self, name: str) -> Optional[HealthStatus]:
        """
        Get the current health status of a provider.

        Parameters
        ----------
        name : str
            Name of the provider.

        Returns
        -------
        HealthStatus or None
            Current health status, or None if provider not registered.
        """
        with self._lock:
            if name in self._providers:
                return self._providers[name].status
            return None

    def get_all_statuses(self) -> Dict[str, HealthStatus]:
        """
        Get health status of all registered providers.

        Returns
        -------
        dict
            Mapping of provider names to their health status.
        """
        with self._lock:
            return {name: p.status for name, p in self._providers.items()}

    def _update_status_metrics(self, name: str, status: HealthStatus) -> None:
        """Update Prometheus status metrics for a provider."""
        provider = self._providers.get(name)
        if not provider:
            return

        provider_type = provider.metadata.get("type", "unknown")
        provider_category = provider.metadata.get("category", "unknown")

        if status == HealthStatus.HEALTHY:
            self._status_gauge.labels(
                provider=name,
                type=provider_type,
                category=provider_category,
                status="healthy",
            ).set(1)
            self._status_gauge.labels(
                provider=name,
                type=provider_type,
                category=provider_category,
                status="unhealthy",
            ).set(0)
        else:
            self._status_gauge.labels(
                provider=name,
                type=provider_type,
                category=provider_category,
                status="healthy",
            ).set(0)
            self._status_gauge.labels(
                provider=name,
                type=provider_type,
                category=provider_category,
                status="unhealthy",
            ).set(1)

    def _health_check_loop(self) -> None:
        """Background thread that checks provider health periodically."""
        while self._running:
            try:
                self._perform_health_check()
            except Exception as e:
                logging.error(f"Error in health check loop: {e}")
            time.sleep(self._check_interval)

    def _perform_health_check(self) -> None:
        """Check health of all providers and update metrics."""
        current_time = time.time()

        # Update uptime
        self._uptime_gauge.set(current_time - self._start_time)

        healthy_count = 0
        unhealthy_count = 0

        with self._lock:
            providers_copy = list(self._providers.items())

        for name, provider in providers_copy:
            seconds_since_heartbeat = current_time - provider.last_heartbeat
            provider_type = provider.metadata.get("type", "unknown")
            provider_category = provider.metadata.get("category", "unknown")
            self._heartbeat_gauge.labels(
                provider=name, type=provider_type, category=provider_category
            ).set(seconds_since_heartbeat)

            if seconds_since_heartbeat > self._heartbeat_timeout:
                if provider.status == HealthStatus.HEALTHY:
                    logging.warning(
                        f"Provider {name} unhealthy: no heartbeat for "
                        f"{seconds_since_heartbeat:.1f}s"
                    )
                    with self._lock:
                        if name in self._providers:
                            self._providers[name].status = HealthStatus.UNHEALTHY
                            self._update_status_metrics(name, HealthStatus.UNHEALTHY)
                unhealthy_count += 1
            else:
                healthy_count += 1

        # Update totals
        self._providers_total.labels(status="healthy").set(healthy_count)
        self._providers_total.labels(status="unhealthy").set(unhealthy_count)

