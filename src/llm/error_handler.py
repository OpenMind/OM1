"""
Error handling utilities for LLM API calls.

This module provides centralized error handling for API errors,
particularly for handling billing-related errors (HTTP 402).
"""

import logging
from typing import Optional

import openai


def handle_llm_api_error(error: Exception, provider_name: str) -> None:
    """
    Handle LLM API errors with specific handling for billing errors.

    This function provides user-friendly error messages for common API errors,
    especially HTTP 402 (Payment Required / Insufficient Balance) errors.

    Parameters
    ----------
    error : Exception
        The exception raised by the API call.
    provider_name : str
        Name of the LLM provider for logging purposes.
    """
    if isinstance(error, openai.APIStatusError):
        status_code = error.status_code

        if status_code == 402:
            logging.error(
                f"{provider_name} API error: Insufficient balance (HTTP 402). "
                "Please check your account balance at https://portal.openmind.org/"
            )
            logging.warning(
                "Your OpenMind account balance may be low. "
                "Consider adding credits to continue using the service."
            )
        elif status_code == 401:
            logging.error(
                f"{provider_name} API error: Authentication failed (HTTP 401). "
                "Please check your API key configuration."
            )
        elif status_code == 403:
            logging.error(
                f"{provider_name} API error: Permission denied (HTTP 403). "
                "Your API key may not have access to this resource."
            )
        elif status_code == 429:
            logging.error(
                f"{provider_name} API error: Rate limit exceeded (HTTP 429). "
                "Please wait before making more requests."
            )
        else:
            logging.error(f"{provider_name} API error (HTTP {status_code}): {error}")
    else:
        logging.error(f"{provider_name} API error: {error}")


def is_insufficient_balance_error(error: Exception) -> bool:
    """
    Check if the error is an insufficient balance error (HTTP 402).

    Parameters
    ----------
    error : Exception
        The exception to check.

    Returns
    -------
    bool
        True if the error is an HTTP 402 error, False otherwise.
    """
    if isinstance(error, openai.APIStatusError):
        return error.status_code == 402
    return False
