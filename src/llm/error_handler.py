"""
Centralized error handling for LLM API calls.

This module provides utilities for handling common API errors across
different LLM providers, including gateway errors (502, 503, 504),
authentication errors, and rate limiting.
"""

import asyncio
import logging
import typing as T
from functools import wraps

import openai


def handle_llm_api_error(error: Exception, provider_name: str) -> None:
    """
    Handle LLM API errors with user-friendly messages.

    Parameters
    ----------
    error : Exception
        The exception that was raised.
    provider_name : str
        The name of the LLM provider (e.g., "OpenAI", "Gemini").
    """
    if isinstance(error, openai.APIStatusError):
        status_code = error.status_code

        if status_code == 401:
            logging.error(
                f"{provider_name} API error: Invalid API key (HTTP 401). "
                "Please check your API key configuration."
            )
        elif status_code == 402:
            logging.error(
                f"{provider_name} API error: Insufficient balance (HTTP 402). "
                "Please check your account balance at https://portal.openmind.org/"
            )
        elif status_code == 403:
            logging.error(
                f"{provider_name} API error: Access forbidden (HTTP 403). "
                "Please check your API permissions."
            )
        elif status_code == 429:
            logging.error(
                f"{provider_name} API error: Rate limit exceeded (HTTP 429). "
                "Please wait before making more requests."
            )
        elif status_code == 502:
            logging.warning(
                f"{provider_name} API error: Bad Gateway (HTTP 502). "
                "The upstream server returned an invalid response. This is usually temporary."
            )
        elif status_code == 503:
            logging.warning(
                f"{provider_name} API error: Service Unavailable (HTTP 503). "
                "The service is temporarily overloaded or under maintenance."
            )
        elif status_code == 504:
            logging.warning(
                f"{provider_name} API error: Gateway Timeout (HTTP 504). "
                "The upstream server did not respond in time."
            )
        else:
            logging.error(f"{provider_name} API error (HTTP {status_code}): {error}")
    elif isinstance(error, openai.APIConnectionError):
        logging.warning(
            f"{provider_name} API connection error: {error}. "
            "Please check your network connection."
        )
    elif isinstance(error, openai.APITimeoutError):
        logging.warning(
            f"{provider_name} API timeout: The request timed out. "
            "This may be due to high server load."
        )
    else:
        logging.error(f"{provider_name} API error: {error}")


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable (gateway errors, timeouts, connection errors).

    Parameters
    ----------
    error : Exception
        The exception to check.

    Returns
    -------
    bool
        True if the error is retryable, False otherwise.
    """
    if isinstance(error, openai.APIStatusError):
        # Gateway errors are retryable
        return error.status_code in (502, 503, 504)
    elif isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
        # Connection and timeout errors are retryable
        return True
    return False


def with_retry(
    max_retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    provider_name: str = "LLM",
) -> T.Callable:
    """
    Decorator for adding retry logic with exponential backoff to async functions.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts (default: 2).
    base_delay : float
        Base delay in seconds between retries (default: 1.0).
    max_delay : float
        Maximum delay in seconds between retries (default: 10.0).
    provider_name : str
        Name of the provider for logging purposes.

    Returns
    -------
    Callable
        Decorated function with retry logic.
    """

    def decorator(func: T.Callable) -> T.Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not is_retryable_error(e):
                        # Non-retryable error, log and return immediately
                        handle_llm_api_error(e, provider_name)
                        return None

                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (2**attempt), max_delay)
                        logging.warning(
                            f"{provider_name}: Retryable error occurred (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        handle_llm_api_error(e, provider_name)
                        await asyncio.sleep(delay)
                    else:
                        # Final attempt failed
                        logging.error(
                            f"{provider_name}: All {max_retries + 1} attempts failed."
                        )
                        handle_llm_api_error(e, provider_name)

            return None

        return wrapper

    return decorator
