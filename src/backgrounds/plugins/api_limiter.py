"""
API Limiter Background
限制 API 调用速率以控制成本
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime

from backgrounds.base import Background, BackgroundConfig


class APILimiter(Background):
    """
    Rate limiter for API calls to control costs and prevent throttling.
    
    Tracks API usage across the system and enforces rate limits:
    - Requests per time window
    - Cost tracking
    - Automatic throttling
    - Usage statistics
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        # Rate limiting configuration
        self.max_requests_per_minute = getattr(config, "max_requests_per_minute", 60)
        self.max_cost_per_hour = getattr(config, "max_cost_per_hour", 1.0)  # USD

        # Cost tracking (approximate costs per request)
        self.cost_per_request = getattr(config, "cost_per_request", 0.001)

        self._running = True
        self.request_times = deque()
        self.total_cost = 0.0
        self.hour_start_time = time.time()
        self.lock = threading.Lock()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True
        )
        self._cleanup_thread.start()

        logging.info(
            f"✅ APILimiter: Started "
            f"(max {self.max_requests_per_minute}/min, "
            f"${self.max_cost_per_hour}/hr)"
        )

    def can_make_request(self) -> bool:
        """
        Check if a request is allowed under rate limits.
        
        Returns
        -------
        bool
            True if request is allowed, False otherwise
        """
        with self.lock:
            current_time = time.time()

            # Clean old timestamps
            cutoff = current_time - 60
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()

            # Check rate limit
            if len(self.request_times) >= self.max_requests_per_minute:
                return False

            # Check cost limit
            if self.total_cost >= self.max_cost_per_hour:
                return False

            # Reset hourly cost if needed
            if current_time - self.hour_start_time >= 3600:
                self.total_cost = 0.0
                self.hour_start_time = current_time

            return True

    def record_request(self, cost: float = None):
        """
        Record a completed API request.
        
        Parameters
        ----------
        cost : float, optional
            Actual cost of the request. If not provided, uses default cost_per_request.
        """
        with self.lock:
            current_time = time.time()

            # Reset hourly cost if needed
            if current_time - self.hour_start_time >= 3600:
                self.total_cost = 0.0
                self.hour_start_time = current_time

            # Record request
            self.request_times.append(current_time)
            actual_cost = cost if cost is not None else self.cost_per_request
            self.total_cost += actual_cost

    def get_usage_stats(self) -> dict:
        """Get current API usage statistics."""
        with self.lock:
            current_time = time.time()
            cutoff = current_time - 60

            # Count requests in last minute
            recent_requests = sum(
                1 for t in self.request_times if t >= cutoff
            )

            # Calculate elapsed time in current hour
            hour_elapsed = current_time - self.hour_start_time

            return {
                "requests_last_minute": recent_requests,
                "requests_per_minute_limit": self.max_requests_per_minute,
                "cost_this_hour": self.total_cost,
                "cost_per_hour_limit": self.max_cost_per_hour,
                "hour_elapsed_seconds": hour_elapsed,
                "utilization_percent": (
                    recent_requests / self.max_requests_per_minute * 100
                    if self.max_requests_per_minute > 0
                    else 0
                ),
            }

    def _cleanup_loop(self):
        """Periodically clean old request timestamps."""
        while self._running:
            try:
                time.sleep(60)
                with self.lock:
                    current_time = time.time()
                    cutoff = current_time - 60
                    while self.request_times and self.request_times[0] < cutoff:
                        self.request_times.popleft()

                    # Reset hourly cost
                    if current_time - self.hour_start_time >= 3600:
                        self.total_cost = 0.0
                        self.hour_start_time = current_time

            except Exception as e:
                logging.error(f"APILimiter cleanup error: {e}")
