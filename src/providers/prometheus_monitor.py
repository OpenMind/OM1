"""Prometheus-based health monitoring for OM1 providers."""

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
    recovery_callback: Optional[Callable[[], bool]] = None
    last_heartbeat: float = 0.0
    error_count: int = 0
    status: HealthStatus = HealthStatus.UNKNOWN


@singleton
class PrometheusMonitor:
    """
    Prometheus-based health monitoring singleton for OM1 providers.

    Provides metrics collection, heartbeat tracking, error counting,
    and automatic recovery for registered providers.

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
            return Response(
                content=generate_latest(), media_type=CONTENT_TYPE_LATEST
            )

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
                providers_data.append({
                    "name": name,
                    "type": provider.metadata.get("type", "unknown"),
                    "category": provider.metadata.get("category", "unknown"),
                    "status": provider.status.value,
                    "last_heartbeat": seconds_since_heartbeat,
                    "error_count": provider.error_count,
                })

        healthy = sum(1 for p in providers_data if p["status"] == "healthy")
        unhealthy = len(providers_data) - healthy

        return {
            "uptime": current_time - self._start_time,
            "healthy_count": healthy,
            "unhealthy_count": unhealthy,
            "providers": providers_data,
        }

    def _generate_dashboard_html(self) -> str:
        """Generate HTML dashboard for health monitoring."""
        data = self._get_health_data()
        uptime_min = int(data["uptime"] / 60)
        uptime_sec = int(data["uptime"] % 60)

        # Group providers by category
        categories: Dict[str, list] = {}
        for p in data["providers"]:
            cat = p["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(p)

        # Generate provider rows
        provider_html = ""
        for category in sorted(categories.keys()):
            providers = categories[category]
            provider_html += f"""
            <div class="category-section">
                <h3 class="category-title">{category.upper()}</h3>
                <div class="provider-grid">
            """
            for p in providers:
                status_class = "healthy" if p["status"] == "healthy" else "unhealthy"
                heartbeat = f"{p['last_heartbeat']:.1f}s"
                provider_html += f"""
                <div class="provider-card {status_class}">
                    <div class="provider-name">{p['name']}</div>
                    <div class="provider-type">{p['type']}</div>
                    <div class="provider-stats">
                        <span class="heartbeat">Heartbeat: {heartbeat}</span>
                        <span class="errors">Errors: {p['error_count']}</span>
                    </div>
                </div>
                """
            provider_html += "</div></div>"

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>OM1 Health Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 20px 0 30px;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.5rem; color: #00d9ff; margin-bottom: 10px; }}
        .header .subtitle {{ color: #888; font-size: 1rem; }}
        .stats-row {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-bottom: 40px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px 40px;
            text-align: center;
        }}
        .stat-value {{ font-size: 3rem; font-weight: bold; }}
        .stat-value.healthy {{ color: #00ff88; }}
        .stat-value.unhealthy {{ color: #ff4757; }}
        .stat-value.uptime {{ color: #00d9ff; }}
        .stat-label {{ color: #888; margin-top: 5px; }}
        .category-section {{ margin-bottom: 30px; }}
        .category-title {{
            font-size: 1.2rem;
            color: #00d9ff;
            margin-bottom: 15px;
            padding-left: 10px;
            border-left: 3px solid #00d9ff;
        }}
        .provider-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }}
        .provider-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px 20px;
            border-left: 4px solid #00ff88;
            transition: transform 0.2s;
        }}
        .provider-card:hover {{ transform: translateX(5px); }}
        .provider-card.unhealthy {{ border-left-color: #ff4757; }}
        .provider-name {{ font-weight: bold; font-size: 1.1rem; margin-bottom: 5px; }}
        .provider-type {{ color: #888; font-size: 0.85rem; margin-bottom: 10px; }}
        .provider-stats {{ display: flex; gap: 20px; font-size: 0.85rem; }}
        .provider-stats span {{ color: #aaa; }}
        .footer {{
            text-align: center;
            padding: 30px 0;
            color: #555;
            font-size: 0.85rem;
        }}
        .footer a {{ color: #00d9ff; text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OM1 Health Dashboard</h1>
            <div class="subtitle">Real-time provider health monitoring</div>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value healthy">{data['healthy_count']}</div>
                <div class="stat-label">Healthy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value unhealthy">{data['unhealthy_count']}</div>
                <div class="stat-label">Unhealthy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value uptime">{uptime_min}m {uptime_sec}s</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>

        {provider_html}

        <div class="footer">
            <a href="/metrics">View Raw Metrics</a> |
            <a href="/api/health">Health API (JSON)</a>
        </div>
    </div>
</body>
</html>
"""

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
            Function to call when attempting recovery. Should return True on success.
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
                recovery_callback=recovery_callback,
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

                    # Attempt recovery
                    self._attempt_recovery(name, provider)
                unhealthy_count += 1
            else:
                healthy_count += 1

        # Update totals
        self._providers_total.labels(status="healthy").set(healthy_count)
        self._providers_total.labels(status="unhealthy").set(unhealthy_count)

    def _attempt_recovery(self, name: str, provider: ProviderState) -> None:
        """
        Attempt to recover an unhealthy provider.

        Parameters
        ----------
        name : str
            Name of the provider.
        provider : ProviderState
            Provider state object.
        """
        if provider.recovery_callback is None:
            return

        logging.info(f"Attempting recovery for provider: {name}")
        try:
            success = provider.recovery_callback()
            if success:
                with self._lock:
                    if name in self._providers:
                        self._providers[name].status = HealthStatus.HEALTHY
                        self._providers[name].last_heartbeat = time.time()
                        self._update_status_metrics(name, HealthStatus.HEALTHY)
                logging.info(f"Recovery successful for provider: {name}")
            else:
                logging.error(f"Recovery failed for provider: {name}")
        except Exception as e:
            logging.error(f"Recovery exception for provider {name}: {e}")
