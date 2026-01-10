"""Prometheus-based health monitoring for OM1 providers."""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from prometheus_client import Counter, Gauge, start_http_server

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
        ["provider", "status"],
    )
    _heartbeat_gauge = Gauge(
        "om1_provider_seconds_since_heartbeat",
        "Seconds since last heartbeat from provider",
        ["provider"],
    )
    _error_counter = Counter(
        "om1_provider_errors_total",
        "Total errors reported by provider",
        ["provider"],
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

        # Reference module-level metrics
        self._status_gauge = _status_gauge
        self._heartbeat_gauge = _heartbeat_gauge
        self._error_counter = _error_counter
        self._providers_total = _providers_total
        self._uptime_gauge = _uptime_gauge

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
                    start_http_server(port)
                    self._server_started = True
                    logging.info(f"Prometheus metrics server started on port {port}")
                except OSError as e:
                    logging.warning(f"Could not start metrics server: {e}")

            if not self._running:
                self._running = True
                self._check_thread = threading.Thread(
                    target=self._health_check_loop, daemon=True
                )
                self._check_thread.start()
                logging.info("Health check thread started")

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

            self._providers[name] = ProviderState(
                name=name,
                metadata=metadata or {},
                recovery_callback=recovery_callback,
                last_heartbeat=time.time(),
                status=HealthStatus.HEALTHY,
            )

            # Initialize metrics for this provider
            self._status_gauge.labels(provider=name, status="healthy").set(1)
            self._status_gauge.labels(provider=name, status="unhealthy").set(0)
            self._heartbeat_gauge.labels(provider=name).set(0)

            logging.info(f"Registered provider: {name}")

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
                self._error_counter.labels(provider=name).inc()
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
        if status == HealthStatus.HEALTHY:
            self._status_gauge.labels(provider=name, status="healthy").set(1)
            self._status_gauge.labels(provider=name, status="unhealthy").set(0)
        else:
            self._status_gauge.labels(provider=name, status="healthy").set(0)
            self._status_gauge.labels(provider=name, status="unhealthy").set(1)

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
            self._heartbeat_gauge.labels(provider=name).set(seconds_since_heartbeat)

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
