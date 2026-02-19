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
        mock_server = MagicMock()
        mock_cls = MagicMock(return_value=mock_server)
        with patch("runtime.metrics.HTTPServer", mock_cls):
            start_metrics_server()
        mock_cls.assert_called_once()
        assert mock_cls.call_args[0][0] == ("", 9464)
        mock_server.serve_forever.assert_called_once()

    def test_custom_port_from_env(self):
        mock_server = MagicMock()
        mock_cls = MagicMock(return_value=mock_server)
        with (
            patch("runtime.metrics.HTTPServer", mock_cls),
            patch.dict("os.environ", {"METRICS_PORT": "8888"}),
        ):
            start_metrics_server()
        assert mock_cls.call_args[0][0] == ("", 8888)

    def test_invalid_port_env_uses_default(self):
        mock_server = MagicMock()
        mock_cls = MagicMock(return_value=mock_server)
        with (
            patch("runtime.metrics.HTTPServer", mock_cls),
            patch.dict("os.environ", {"METRICS_PORT": "not_a_number"}),
        ):
            start_metrics_server()
        assert mock_cls.call_args[0][0] == ("", 9464)

    def test_port_in_use_logs_warning_not_crash(self):
        mock_cls = MagicMock(side_effect=OSError("Address already in use"))
        with patch("runtime.metrics.HTTPServer", mock_cls):
            # Should not raise
            start_metrics_server()


class TestRenderDashboard:
    """Test the HTML dashboard renderer."""

    def test_empty_dashboard(self):
        from runtime.metrics import _render_dashboard

        html = _render_dashboard()
        assert "<!DOCTYPE html>" in html
        assert "OM1 Metrics Dashboard" in html
        assert "content='5'" in html

    def test_dashboard_shows_status_dots(self):
        from runtime.metrics import _render_dashboard

        INPUT_STATUS.labels(name="cam").set(1)
        INPUT_STATUS.labels(name="mic").set(-1)
        INPUT_STATUS.labels(name="lidar").set(0)

        html = _render_dashboard()
        assert "dot green" in html
        assert "running" in html
        assert "dot red" in html
        assert "failed" in html
        assert "dot gray" in html
        assert "stopped" in html

    def test_dashboard_shows_counter_values(self):
        from runtime.metrics import _render_dashboard

        INPUT_EVENTS_TOTAL.labels(name="cam").inc()

        html = _render_dashboard()
        assert "cam" in html
        assert "<h2>Inputs</h2>" in html

    def test_dashboard_shows_histogram_count_and_sum(self):
        from runtime.metrics import _render_dashboard

        CORTEX_TICK_DURATION.observe(0.5)

        html = _render_dashboard()
        assert "Tick Duration (s) Count" in html
        assert "Tick Duration (s) Sum" in html

    def test_dashboard_filters_python_metrics(self):
        from runtime.metrics import _render_dashboard

        html = _render_dashboard()
        assert "python_gc" not in html
        assert "python_info" not in html

    def test_dashboard_shows_all_categories(self):
        from runtime.metrics import _render_dashboard

        INPUT_STATUS.labels(name="x").set(1)
        ACTION_STATUS.labels(name="x").set(1)
        BACKGROUND_STATUS.labels(name="x").set(1)
        SIMULATOR_STATUS.labels(name="x").set(1)
        CORTEX_TICKS_TOTAL.inc()
        MODE_CURRENT.labels(mode="x").set(1)

        html = _render_dashboard()
        for cat in ["Inputs", "Actions", "Backgrounds", "Simulators", "Cortex", "Mode"]:
            assert f"<h2>{cat}</h2>" in html

    def test_dashboard_formats_float_values(self):
        from runtime.metrics import _render_dashboard

        CORTEX_TICK_DURATION.observe(0.1234)

        html = _render_dashboard()
        assert "0.1234" in html or "Tick Duration" in html


class TestMetricsHandler:
    """Test the custom HTTP handler via a real server."""

    def _start_server(self):
        import threading
        from http.server import HTTPServer

        from runtime.metrics import _MetricsHandler

        server = HTTPServer(("127.0.0.1", 0), _MetricsHandler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        return server, port

    def test_metrics_endpoint(self):
        import urllib.request

        server, port = self._start_server()
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics")
            assert resp.status == 200
            assert "text/plain" in resp.headers["Content-Type"]
            body = resp.read().decode()
            assert "om1_" in body
        finally:
            server.shutdown()

    def test_dashboard_endpoint(self):
        import urllib.request

        server, port = self._start_server()
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
            assert resp.status == 200
            assert "text/html" in resp.headers["Content-Type"]
            body = resp.read().decode()
            assert "OM1 Metrics Dashboard" in body
        finally:
            server.shutdown()

    def test_404_for_unknown_path(self):
        import urllib.error
        import urllib.request

        server, port = self._start_server()
        try:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/unknown")
                assert False, "Expected 404"
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            server.shutdown()


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
