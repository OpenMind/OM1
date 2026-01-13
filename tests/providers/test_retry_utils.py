"""Tests for retry utilities with exponential backoff."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from providers.retry_utils import (
    RetryError,
    retry_on_http_error,
    retry_with_exponential_backoff,
)


class TestSyncRetry:
    """Test synchronous retry functionality."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_exception(self):
        """Test retry on exception."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 3

    def test_max_attempts_exceeded(self):
        """Test that RetryError is raised after max attempts."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            always_failing_func()

        assert call_count == 3
        assert "failed after 3 attempts" in str(exc_info.value)
        assert exc_info.value.last_exception is not None

    def test_exponential_backoff_delay(self):
        """Test that delays increase exponentially."""
        call_times = []

        @retry_with_exponential_backoff(
            max_attempts=3, initial_delay=0.1, exponential_base=2.0, jitter=False
        )
        def failing_func():
            call_times.append(time.time())
            raise ValueError("Fail")

        start_time = time.time()
        with pytest.raises(RetryError):
            failing_func()

        # Check that delays increase exponentially
        delays = [
            call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)
        ]
        # Allow some tolerance for timing
        assert delays[0] >= 0.1 * 0.9  # First delay ~0.1s
        assert delays[1] >= 0.2 * 0.9  # Second delay ~0.2s (0.1 * 2^1)

    def test_jitter_added(self):
        """Test that jitter is added to delays."""
        delays = []

        @retry_with_exponential_backoff(
            max_attempts=3, initial_delay=0.1, jitter=True
        )
        def failing_func():
            delays.append(time.time())
            raise ValueError("Fail")

        start_time = time.time()
        with pytest.raises(RetryError):
            failing_func()

        # With jitter, delays should vary slightly
        if len(delays) >= 2:
            delay = delays[1] - delays[0]
            # Should be at least base delay, but can be up to 25% more
            assert delay >= 0.1
            assert delay <= 0.125 * 1.1  # Allow some tolerance

    def test_retryable_exceptions_filter(self):
        """Test that only retryable exceptions trigger retries."""
        call_count = 0

        @retry_with_exponential_backoff(
            max_attempts=3,
            retryable_exceptions=(ValueError,),
            initial_delay=0.01,
        )
        def func_with_different_exception():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            func_with_different_exception()

        assert call_count == 1  # Should not retry

    def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        call_count = 0
        retry_calls = []

        def on_retry(attempt: int, exc: Exception):
            retry_calls.append((attempt, str(exc)))

        @retry_with_exponential_backoff(
            max_attempts=3, initial_delay=0.1, on_retry=on_retry
        )
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        result = failing_func()
        assert result == "success"
        assert len(retry_calls) == 1
        assert retry_calls[0][0] == 1  # First retry attempt

    def test_reraise_false(self):
        """Test that reraise=False returns None instead of raising."""
        call_count = 0

        @retry_with_exponential_backoff(
            max_attempts=2, initial_delay=0.1, reraise=False
        )
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        result = always_failing_func()
        assert result is None
        assert call_count == 2


class TestAsyncRetry:
    """Test asynchronous retry functionality."""

    @pytest.mark.asyncio
    async def test_async_successful_call_no_retry(self):
        """Test that successful async calls don't retry."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_exception(self):
        """Test retry on async exception."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await failing_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_max_attempts_exceeded(self):
        """Test that RetryError is raised after max async attempts."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)
        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(RetryError):
            await always_failing_func()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_exponential_backoff_delay(self):
        """Test that async delays increase exponentially."""
        call_times = []

        @retry_with_exponential_backoff(
            max_attempts=3, initial_delay=0.1, exponential_base=2.0, jitter=False
        )
        async def failing_func():
            call_times.append(time.time())
            raise ValueError("Fail")

        start_time = time.time()
        with pytest.raises(RetryError):
            await failing_func()

        # Check that delays increase exponentially
        if len(call_times) >= 2:
            delays = [
                call_times[i + 1] - call_times[i]
                for i in range(len(call_times) - 1)
            ]
            assert delays[0] >= 0.1 * 0.9
            assert delays[1] >= 0.2 * 0.9


class TestHttpRetry:
    """Test HTTP-specific retry functionality."""

    @patch("requests.get")
    def test_retry_on_http_error_5xx(self, mock_get):
        """Test retry on 5xx HTTP errors."""
        call_count = 0

        @retry_on_http_error(max_attempts=3, initial_delay=0.1)
        def fetch_data():
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            if call_count < 3:
                mock_response.status_code = 500
                mock_response.raise_for_status.side_effect = requests.HTTPError(
                    "Server Error"
                )
            else:
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": "success"}
            mock_get.return_value = mock_response
            response = requests.get("https://api.example.com/data")
            response.raise_for_status()
            return response.json()

        result = fetch_data()
        assert result == {"data": "success"}
        assert call_count == 3

    @patch("requests.get")
    def test_no_retry_on_4xx_error(self, mock_get):
        """Test that 4xx errors don't trigger retries."""
        call_count = 0

        @retry_on_http_error(max_attempts=3, initial_delay=0.1)
        def fetch_data():
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "Not Found"
            )
            mock_get.return_value = mock_response
            response = requests.get("https://api.example.com/data")
            response.raise_for_status()
            return response.json()

        with pytest.raises(requests.HTTPError):
            fetch_data()

        assert call_count == 1  # Should not retry on 4xx

    @patch("requests.get")
    def test_custom_status_codes(self, mock_get):
        """Test retry on custom status codes."""
        call_count = 0

        @retry_on_http_error(
            max_attempts=3, initial_delay=0.1, status_codes=[429, 503]
        )
        def fetch_data():
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            if call_count < 2:
                mock_response.status_code = 429
                mock_response.raise_for_status.side_effect = requests.HTTPError(
                    "Too Many Requests"
                )
            else:
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": "success"}
            mock_get.return_value = mock_response
            response = requests.get("https://api.example.com/data")
            response.raise_for_status()
            return response.json()

        result = fetch_data()
        assert result == {"data": "success"}
        assert call_count == 2


class TestRetryEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_max_attempts(self):
        """Test behavior with zero max attempts."""
        call_count = 0

        @retry_with_exponential_backoff(max_attempts=0, initial_delay=0.1)
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Fail")

        with pytest.raises(RetryError):
            failing_func()

        assert call_count == 0  # Should not even try

    def test_very_small_delay(self):
        """Test with very small delay values."""
        call_count = 0

        @retry_with_exponential_backoff(
            max_attempts=3, initial_delay=0.001, max_delay=0.01
        )
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count == 2

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        call_times = []

        @retry_with_exponential_backoff(
            max_attempts=4,
            initial_delay=10.0,
            max_delay=0.5,
            exponential_base=2.0,
            jitter=False,
        )
        def failing_func():
            call_times.append(time.time())
            raise ValueError("Fail")

        with pytest.raises(RetryError):
            failing_func()

        if len(call_times) >= 2:
            delays = [
                call_times[i + 1] - call_times[i]
                for i in range(len(call_times) - 1)
            ]
            # All delays should be capped at max_delay
            for delay in delays:
                assert delay <= 0.5 * 1.1  # Allow some tolerance
