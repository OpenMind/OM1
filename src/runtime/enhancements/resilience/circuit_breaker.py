"""Circuit Breaker pattern implementation for resilient external service calls."""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit is open, calls are failing
    HALF_OPEN = "HALF_OPEN"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Optional[type] = None,
        name: str = "CircuitBreaker"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception or Exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"circuit_breaker.{name}")
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function call through the circuit breaker."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self._logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    async def _on_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self._logger.info(f"Circuit breaker {self.name} reset to CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    async def _on_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self._logger.warning(
                    f"Circuit breaker {self.name} opened after {self.failure_count} failures"
                )
    
    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN
    
    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED
    
    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN
    
    def reset(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self._logger.info(f"Circuit breaker {self.name} manually reset")


class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


async def retry_with_exponential_backoff(
    func: Callable[..., T],
    retry_config: RetryConfig,
    *args,
    **kwargs
) -> T:
    last_exception = None
    
    for attempt in range(retry_config.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if attempt == retry_config.max_attempts - 1:
                break
            
            delay = min(
                retry_config.base_delay * (retry_config.exponential_base ** attempt),
                retry_config.max_delay
            )
            
            if retry_config.jitter:
                import random
                delay *= (0.5 + random.random() * 0.5)
            
            logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    
    raise last_exception
