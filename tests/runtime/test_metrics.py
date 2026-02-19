import sys
from unittest.mock import MagicMock, patch

from prometheus_client import Counter, Gauge, Histogram

from runtime.metrics import (
    ACTION_ERRORS_TOTAL,
    ACTION_EXECUTIONS_TOTAL,
    ACTION_STATUS,
    BACKGROUND_ERRORS_TOTAL,
    BACKGROUND_RUNS_TOTAL,
    BACKGROUND_STATUS,
    CORTEX_TICK_DURATION,
    CORTEX_TICKS_TOTAL,
    INPUT_ERRORS_TOTAL,
    INPUT_EVENTS_TOTAL,
    INPUT_STATUS,
    MODE_CURRENT,
    MODE_TRANSITIONS_TOTAL,
    SIMULATOR_ERRORS_TOTAL,
    SIMULATOR_STATUS,
    SIMULATOR_TICKS_TOTAL,
    _NoOp,
    start_metrics_server,
)


class TestNoOp:
    """Test that the _NoOp stub silently ignores all metric operations."""

    def test_labels_returns_self(self):
        noop = _NoOp()
        result = noop.labels(name="test")
        assert result is noop

    def test_labels_chaining(self):
        noop = _NoOp()
        result = noop.labels(name="test").labels(foo="bar")
        assert result is noop

    def test_inc_returns_none(self):
        noop = _NoOp()
        assert noop.inc() is None
        assert noop.inc(5) is None

    def test_set_returns_none(self):
        noop = _NoOp()
        assert noop.set(1) is None
        assert noop.set(-1) is None

    def test_observe_returns_none(self):
        noop = _NoOp()
        assert noop.observe(0.5) is None

    def test_labels_then_inc(self):
        """Test the full chain: metric.labels(name=x).inc() -- the real usage pattern."""
        noop = _NoOp()
        noop.labels(name="my_input").inc()

    def test_labels_then_set(self):
        noop = _NoOp()
        noop.labels(name="my_input").set(1)

    def test_labels_then_observe(self):
        noop = _NoOp()
        noop.labels(name="my_input").observe(0.123)


class TestMetricTypes:
    """Test that metrics are real prometheus types when the library is installed."""

    def test_gauges_are_gauge_type(self):
        assert isinstance(INPUT_STATUS, Gauge)
        assert isinstance(ACTION_STATUS, Gauge)
        assert isinstance(BACKGROUND_STATUS, Gauge)
        assert isinstance(SIMULATOR_STATUS, Gauge)
        assert isinstance(MODE_CURRENT, Gauge)

    def test_counters_are_counter_type(self):
        assert isinstance(INPUT_EVENTS_TOTAL, Counter)
        assert isinstance(INPUT_ERRORS_TOTAL, Counter)
        assert isinstance(ACTION_EXECUTIONS_TOTAL, Counter)
        assert isinstance(ACTION_ERRORS_TOTAL, Counter)
        assert isinstance(BACKGROUND_RUNS_TOTAL, Counter)
        assert isinstance(BACKGROUND_ERRORS_TOTAL, Counter)
        assert isinstance(SIMULATOR_TICKS_TOTAL, Counter)
        assert isinstance(SIMULATOR_ERRORS_TOTAL, Counter)
        assert isinstance(CORTEX_TICKS_TOTAL, Counter)
        assert isinstance(MODE_TRANSITIONS_TOTAL, Counter)

    def test_histograms_are_histogram_type(self):
        assert isinstance(CORTEX_TICK_DURATION, Histogram)


class TestMetricOperations:
    """Test that real metric operations work without errors."""

    def test_labeled_gauge_set(self):
        INPUT_STATUS.labels(name="test_sensor").set(1)
        INPUT_STATUS.labels(name="test_sensor").set(-1)

    def test_labeled_counter_inc(self):
        INPUT_EVENTS_TOTAL.labels(name="test_sensor").inc()
        ACTION_EXECUTIONS_TOTAL.labels(name="test_action").inc()

    def test_unlabeled_counter_inc(self):
        CORTEX_TICKS_TOTAL.inc()

    def test_histogram_observe(self):
        CORTEX_TICK_DURATION.observe(0.05)

    def test_mode_gauge_set(self):
        MODE_CURRENT.labels(mode="patrol").set(1)
        MODE_CURRENT.labels(mode="patrol").set(0)

    def test_mode_transition_counter(self):
        MODE_TRANSITIONS_TOTAL.labels(from_mode="idle", to_mode="patrol").inc()


class TestStartMetricsServer:
    """Test start_metrics_server behavior."""

    def test_successful_start(self):
        mock_start = MagicMock()
        with patch("runtime.metrics.start_http_server", mock_start):
            start_metrics_server()
        mock_start.assert_called_once_with(9090)

    def test_custom_port_from_env(self):
        mock_start = MagicMock()
        with (
            patch("runtime.metrics.start_http_server", mock_start),
            patch.dict("os.environ", {"METRICS_PORT": "8888"}),
        ):
            start_metrics_server()
        mock_start.assert_called_once_with(8888)

    def test_port_in_use_logs_warning_not_crash(self):
        mock_start = MagicMock(side_effect=OSError("Address already in use"))
        with patch("runtime.metrics.start_http_server", mock_start):
            # Should not raise
            start_metrics_server()


class TestNoOpFallback:
    """Test that when prometheus_client is missing, the module still loads."""

    def test_module_loads_without_prometheus(self):
        """Simulate prometheus_client not being installed via import patching."""
        saved_metrics = sys.modules.get("runtime.metrics")
        saved_prom_modules = {
            k: sys.modules[k]
            for k in list(sys.modules)
            if k.startswith("prometheus_client")
        }

        # Remove prometheus_client and runtime.metrics from sys.modules
        for key in list(sys.modules):
            if key.startswith("prometheus_client"):
                del sys.modules[key]
        if "runtime.metrics" in sys.modules:
            del sys.modules["runtime.metrics"]

        builtins_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "prometheus_client" or name.startswith("prometheus_client."):
                raise ImportError("mocked: no prometheus_client")
            return builtins_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                import runtime.metrics as m

                # Use the reloaded module's _NoOp (different class identity after reload)
                noop_cls = m._NoOp
                assert isinstance(m.INPUT_STATUS, noop_cls)
                assert isinstance(m.CORTEX_TICKS_TOTAL, noop_cls)
                assert isinstance(m.MODE_CURRENT, noop_cls)

                # All operations should work silently
                m.INPUT_STATUS.labels(name="test").set(1)
                m.CORTEX_TICKS_TOTAL.inc()
                m.CORTEX_TICK_DURATION.observe(0.5)
                m.MODE_TRANSITIONS_TOTAL.labels(from_mode="a", to_mode="b").inc()

                # start_metrics_server should be a no-op
                m.start_metrics_server()
        finally:
            if "runtime.metrics" in sys.modules:
                del sys.modules["runtime.metrics"]
            if saved_metrics is not None:
                sys.modules["runtime.metrics"] = saved_metrics
            sys.modules.update(saved_prom_modules)
