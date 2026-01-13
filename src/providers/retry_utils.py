"""
Retry utilities with exponential backoff for robust API calls and operations.

This module provides decorators and functions for retrying operations with
configurable exponential backoff, exception handling, and rate limiting.
"""

import asyncio
import functools
import logging
import random
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        """
        Initialize RetryError.

        Parameters
        ----------
        message : str
            Error message
        last_exception : Exception, optional
            The last exception that occurred before giving up
        """
        super().__init__(message)
        self.last_exception = last_exception


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    retryable_status_codes: Optional[List[int]] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    reraise: bool = True,
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Supports both synchronous and asynchronous functions.

    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts (default: 3)
    initial_delay : float
        Initial delay in seconds before first retry (default: 1.0)
    max_delay : float
        Maximum delay between retries in seconds (default: 60.0)
    exponential_base : float
        Base for exponential backoff calculation (default: 2.0)
    jitter : bool
        Whether to add random jitter to delays (default: True)
    retryable_exceptions : tuple of Exception classes, optional
        Tuple of exception types that should trigger retries.
        If None, retries on all exceptions.
    retryable_status_codes : list of int, optional
        List of HTTP status codes that should trigger retries.
        Only applicable for HTTP-related exceptions.
    on_retry : callable, optional
        Callback function called on each retry attempt.
        Signature: (attempt_number: int, exception: Exception) -> None
    reraise : bool
        Whether to reraise the last exception after all retries fail (default: True)

    Returns
    -------
    callable
        Decorated function with retry logic

    Examples
    --------
    >>> @retry_with_exponential_backoff(max_attempts=5, initial_delay=0.5)
    ... def fetch_data():
    ...     response = requests.get("https://api.example.com/data")
    ...     response.raise_for_status()
    ...     return response.json()

    >>> @retry_with_exponential_backoff(
    ...     max_attempts=3,
    ...     retryable_exceptions=(requests.RequestException,),
    ...     on_retry=lambda attempt, exc: print(f"Retry {attempt}: {exc}")
    ... )
    ... async def async_fetch():
    ...     async with aiohttp.ClientSession() as session:
    ...         async with session.get("https://api.example.com/data") as resp:
    ...             return await resp.json()
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            return _async_retry_wrapper(
                func,
                max_attempts,
                initial_delay,
                max_delay,
                exponential_base,
                jitter,
                retryable_exceptions,
                retryable_status_codes,
                on_retry,
                reraise,
            )
        else:
            return _sync_retry_wrapper(
                func,
                max_attempts,
                initial_delay,
                max_delay,
                exponential_base,
                jitter,
                retryable_exceptions,
                retryable_status_codes,
                on_retry,
                reraise,
            )

    return decorator


def _calculate_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
    """
    Calculate delay for retry attempt with exponential backoff.

    Parameters
    ----------
    attempt : int
        Current attempt number (0-indexed)
    initial_delay : float
        Initial delay in seconds
    max_delay : float
        Maximum delay in seconds
    exponential_base : float
        Base for exponential calculation
    jitter : bool
        Whether to add random jitter

    Returns
    -------
    float
        Delay in seconds
    """
    delay = initial_delay * (exponential_base ** attempt)
    delay = min(delay, max_delay)

    if jitter:
        # Add random jitter up to 25% of the delay
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


def _should_retry(
    exception: Exception,
    retryable_exceptions: Tuple[Type[Exception], ...],
    retryable_status_codes: Optional[List[int]],
) -> bool:
    """
    Determine if an exception should trigger a retry.

    Parameters
    ----------
    exception : Exception
        The exception that occurred
    retryable_exceptions : tuple of Exception classes
        Exception types that should trigger retries
    retryable_status_codes : list of int, optional
        HTTP status codes that should trigger retries

    Returns
    -------
    bool
        True if the exception should trigger a retry
    """
    # Check if exception type is retryable
    if not isinstance(exception, retryable_exceptions):
        return False

    # Check HTTP status codes if applicable
    if retryable_status_codes is not None:
        # Try to extract status code from exception
        status_code = getattr(exception, "status_code", None)
        if status_code is not None:
            return status_code in retryable_status_codes

    return True


def _sync_retry_wrapper(
    func: Callable[..., T],
    max_attempts: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
    retryable_exceptions: Tuple[Type[Exception], ...],
    retryable_status_codes: Optional[List[int]],
    on_retry: Optional[Callable[[int, Exception], None]],
    reraise: bool,
) -> Callable[..., T]:
    """Wrap synchronous function with retry logic."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e

                if not _should_retry(e, retryable_exceptions, retryable_status_codes):
                    # Exception is not retryable, reraise immediately
                    raise

                if attempt < max_attempts - 1:
                    delay = _calculate_delay(
                        attempt, initial_delay, max_delay, exponential_base, jitter
                    )

                    if on_retry:
                        try:
                            on_retry(attempt + 1, e)
                        except Exception:
                            pass  # Don't let retry callback break retry logic

                    logging.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)
                else:
                    # Last attempt failed
                    error_msg = (
                        f"{func.__name__} failed after {max_attempts} attempts. "
                        f"Last error: {e}"
                    )
                    logging.error(error_msg)

                    if reraise:
                        raise RetryError(error_msg, last_exception=e) from e
                    else:
                        return None  # type: ignore

        # Should never reach here, but just in case
        if reraise and last_exception:
            raise RetryError(
                f"{func.__name__} failed after {max_attempts} attempts",
                last_exception=last_exception,
            ) from last_exception

        return None  # type: ignore

    return wrapper


def _async_retry_wrapper(
    func: Callable[..., T],
    max_attempts: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
    retryable_exceptions: Tuple[Type[Exception], ...],
    retryable_status_codes: Optional[List[int]],
    on_retry: Optional[Callable[[int, Exception], None]],
    reraise: bool,
) -> Callable[..., T]:
    """Wrap asynchronous function with retry logic."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except retryable_exceptions as e:
                last_exception = e

                if not _should_retry(e, retryable_exceptions, retryable_status_codes):
                    # Exception is not retryable, reraise immediately
                    raise

                if attempt < max_attempts - 1:
                    delay = _calculate_delay(
                        attempt, initial_delay, max_delay, exponential_base, jitter
                    )

                    if on_retry:
                        try:
                            on_retry(attempt + 1, e)
                        except Exception:
                            pass  # Don't let retry callback break retry logic

                    logging.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)
                else:
                    # Last attempt failed
                    error_msg = (
                        f"{func.__name__} failed after {max_attempts} attempts. "
                        f"Last error: {e}"
                    )
                    logging.error(error_msg)

                    if reraise:
                        raise RetryError(error_msg, last_exception=e) from e
                    else:
                        return None  # type: ignore

        # Should never reach here, but just in case
        if reraise and last_exception:
            raise RetryError(
                f"{func.__name__} failed after {max_attempts} attempts",
                last_exception=last_exception,
            ) from last_exception

        return None  # type: ignore

    return wrapper


def retry_on_http_error(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    status_codes: Optional[List[int]] = None,
) -> Callable:
    """
    Convenience decorator for retrying HTTP requests on specific status codes.

    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts (default: 3)
    initial_delay : float
        Initial delay in seconds (default: 1.0)
    status_codes : list of int, optional
        HTTP status codes that should trigger retries.
        If None, retries on 5xx errors only.

    Returns
    -------
    callable
        Decorated function with HTTP retry logic

    Examples
    --------
    >>> @retry_on_http_error(max_attempts=5, status_codes=[500, 502, 503])
    ... def fetch_api_data():
    ...     response = requests.get("https://api.example.com/data")
    ...     response.raise_for_status()
    ...     return response.json()
    """
    if status_codes is None:
        status_codes = [500, 502, 503, 504]

    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests library is required for retry_on_http_error decorator"
        )

    return retry_with_exponential_backoff(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        retryable_exceptions=(requests.RequestException,),
        retryable_status_codes=status_codes,
    )
