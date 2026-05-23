import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from providers.httpx import get_async_httpx_client, get_httpx_event_hooks


class TestGetHttpxEventHooks:
    def test_returns_request_and_response_keys(self):
        hooks = get_httpx_event_hooks()
        assert "request" in hooks
        assert "response" in hooks

    def test_hooks_are_lists(self):
        hooks = get_httpx_event_hooks()
        assert isinstance(hooks["request"], list)
        assert isinstance(hooks["response"], list)

    def test_hooks_are_callable(self):
        hooks = get_httpx_event_hooks()
        assert callable(hooks["request"][0])
        assert callable(hooks["response"][0])

    @pytest.mark.asyncio
    async def test_log_request_sets_start_time(self):
        hooks = get_httpx_event_hooks()
        log_request = hooks["request"][0]

        request = MagicMock(spec=httpx.Request)
        request.extensions = {}

        before = time.perf_counter()
        await log_request(request)
        after = time.perf_counter()

        assert "start_time" in request.extensions
        assert before <= request.extensions["start_time"] <= after

    @pytest.mark.asyncio
    async def test_log_response_logs_info(self):
        hooks = get_httpx_event_hooks()
        log_response = hooks["response"][0]

        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = "https://example.com/api"
        mock_request.extensions = {"start_time": time.perf_counter() - 0.1}

        response = MagicMock(spec=httpx.Response)
        response.request = mock_request
        response.status_code = 200
        response.http_version = "HTTP/2"
        response.headers = {}

        with patch("providers.httpx.logging.info") as mock_log:
            await log_response(response)
            mock_log.assert_called_once()
            log_msg = mock_log.call_args[0][0]
            assert "GET" in log_msg
            assert "https://example.com/api" in log_msg
            assert "200" in log_msg
            assert "HTTP/2" in log_msg

    @pytest.mark.asyncio
    async def test_log_response_includes_proxy_headers(self):
        hooks = get_httpx_event_hooks()
        log_response = hooks["response"][0]

        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "POST"
        mock_request.url = "https://example.com/data"
        mock_request.extensions = {"start_time": time.perf_counter()}

        response = MagicMock(spec=httpx.Response)
        response.request = mock_request
        response.status_code = 201
        response.http_version = "HTTP/1.1"
        response.headers = {
            "x-proxy-parse-ms": "5",
            "x-upstream-total-ms": "120",
            "x-upstream-ttfb-ms": "80",
            "x-proxy-total-ms": "130",
        }

        with patch("providers.httpx.logging.info") as mock_log:
            await log_response(response)
            log_msg = mock_log.call_args[0][0]
            assert "5" in log_msg
            assert "120" in log_msg
            assert "80" in log_msg
            assert "130" in log_msg

    @pytest.mark.asyncio
    async def test_log_response_missing_proxy_headers_uses_question_mark(self):
        hooks = get_httpx_event_hooks()
        log_response = hooks["response"][0]

        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = "https://example.com"
        mock_request.extensions = {}  # no start_time

        response = MagicMock(spec=httpx.Response)
        response.request = mock_request
        response.status_code = 404
        response.http_version = "HTTP/1.1"
        response.headers = {}

        with patch("providers.httpx.logging.info") as mock_log:
            await log_response(response)
            log_msg = mock_log.call_args[0][0]
            assert "?" in log_msg

    @pytest.mark.asyncio
    async def test_log_response_elapsed_uses_zero_when_no_start_time(self):
        hooks = get_httpx_event_hooks()
        log_response = hooks["response"][0]

        mock_request = MagicMock(spec=httpx.Request)
        mock_request.method = "GET"
        mock_request.url = "https://example.com"
        mock_request.extensions = {}

        response = MagicMock(spec=httpx.Response)
        response.request = mock_request
        response.status_code = 200
        response.http_version = "HTTP/1.1"
        response.headers = {}

        with patch("providers.httpx.logging.info") as mock_log:
            await log_response(response)
            mock_log.assert_called_once()


class TestGetAsyncHttpxClient:
    def test_returns_async_client(self):
        client = get_async_httpx_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_default_timeout(self):
        client = get_async_httpx_client()
        assert client.timeout.read == 60.0
        assert client.timeout.connect == 5.0

    def test_custom_timeout(self):
        client = get_async_httpx_client(timeout=30.0, connect_timeout=2.0)
        assert client.timeout.read == 30.0
        assert client.timeout.connect == 2.0

    def test_event_hooks_attached(self):
        client = get_async_httpx_client()
        assert len(client.event_hooks["request"]) == 1
        assert len(client.event_hooks["response"]) == 1

    def test_http2_enabled_by_default(self):
        client = get_async_httpx_client()
        assert client._transport is not None

    def test_http2_can_be_disabled(self):
        client = get_async_httpx_client(http2=False)
        assert isinstance(client, httpx.AsyncClient)

    def test_custom_connection_limits(self):
        client = get_async_httpx_client(
            max_keepalive_connections=5,
            max_connections=10,
            keepalive_expiry=60.0,
        )
        assert isinstance(client, httpx.AsyncClient)
