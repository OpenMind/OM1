"""
Robust JSON extraction and parsing utilities for LLM responses.

This module provides utilities to safely parse JSON from LLM responses
that may be wrapped in Markdown code blocks or contain other text.

Addresses Issue #1700: LLM responses occasionally include Markdown formatting
which causes json.loads() to fail.
"""

import json
import logging
import re
from typing import Any, Optional

# Pattern to match JSON inside Markdown code blocks: ```json ... ``` or ``` ... ```
_MARKDOWN_JSON_PATTERN = re.compile(
    r"```(?:json|JSON)?\s*\n?([\s\S]*?)\n?\s*```",
    re.MULTILINE
)

# Pattern to match a JSON object (starts with { ends with })
_JSON_OBJECT_PATTERN = re.compile(
    r"(\{[\s\S]*\})",
    re.MULTILINE
)

# Pattern to match a JSON array (starts with [ ends with ])
_JSON_ARRAY_PATTERN = re.compile(
    r"(\[[\s\S]*\])",
    re.MULTILINE
)


def extract_json_string(text: str) -> Optional[str]:
    """
    Extract JSON string from text that may contain Markdown formatting.

    Extraction priority:
    1. Content inside Markdown code blocks (```json ... ``` or ``` ... ```)
    2. Raw JSON object ({...})
    3. Raw JSON array ([...])
    4. Original text (if none of the above match)

    Parameters
    ----------
    text : str
        The input text potentially containing JSON

    Returns
    -------
    Optional[str]
        Extracted JSON string, or None if input is invalid

    Examples
    --------
    >>> extract_json_string('```json\\n{"key": "value"}\\n```')
    '{"key": "value"}'
    >>> extract_json_string('Here is the response: {"key": "value"}')
    '{"key": "value"}'
    """
    if not isinstance(text, str) or not text.strip():
        return None

    text = text.strip()

    # Try to extract from Markdown code block first
    markdown_match = _MARKDOWN_JSON_PATTERN.search(text)
    if markdown_match:
        extracted = markdown_match.group(1).strip()
        if extracted:
            return extracted

    # Try to find a JSON object
    obj_match = _JSON_OBJECT_PATTERN.search(text)
    if obj_match:
        return obj_match.group(1)

    # Try to find a JSON array
    arr_match = _JSON_ARRAY_PATTERN.search(text)
    if arr_match:
        return arr_match.group(1)

    # Return original text as fallback
    return text


def safe_json_loads(
    text: str,
    default: Any = None,
    log_errors: bool = True,
    context: str = ""
) -> Any:
    """
    Safely parse JSON from text, with support for Markdown-wrapped responses.

    This function attempts multiple parsing strategies:
    1. Direct JSON parsing
    2. Extract from Markdown code blocks and parse
    3. Find and parse embedded JSON objects/arrays

    Parameters
    ----------
    text : str
        The text to parse as JSON
    default : Any, optional
        Value to return if parsing fails (default: None)
    log_errors : bool, optional
        Whether to log parsing errors (default: True)
    context : str, optional
        Additional context for error messages (default: "")

    Returns
    -------
    Any
        Parsed JSON object, or default value if parsing fails

    Examples
    --------
    >>> safe_json_loads('{"action": "move"}')
    {'action': 'move'}
    >>> safe_json_loads('```json\\n{"action": "move"}\\n```')
    {'action': 'move'}
    >>> safe_json_loads('invalid', default={})
    {}
    """
    if not isinstance(text, str) or not text.strip():
        if log_errors and text is not None:
            logging.debug(f"Empty or invalid input for JSON parsing{_ctx(context)}")
        return default

    # Strategy 1: Try direct parsing first (most common case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Try to extract JSON from the text
    extracted = extract_json_string(text)
    if extracted and extracted != text:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as e:
            if log_errors:
                logging.warning(
                    f"Failed to parse extracted JSON{_ctx(context)}: {e}. "
                    f"Extracted: {_truncate(extracted)}"
                )

    # Strategy 3: Try to find and parse nested JSON
    # This handles cases like: "The response is {"key": "value"} as shown"
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # All strategies failed
    if log_errors:
        logging.error(
            f"Could not parse JSON from text{_ctx(context)}: {_truncate(text)}"
        )

    return default


def safe_parse_function_args(
    args_string: str,
    default: Optional[dict] = None
) -> Optional[dict]:
    """
    Safely parse function call arguments from LLM responses.

    Specifically designed for parsing the 'arguments' field from
    OpenAI-style function calls.

    Parameters
    ----------
    args_string : str
        The arguments string to parse
    default : dict, optional
        Default value if parsing fails (default: None)

    Returns
    -------
    Optional[dict]
        Parsed arguments dictionary, or default if parsing fails
    """
    if default is None:
        default = {}

    result = safe_json_loads(
        args_string,
        default=default,
        log_errors=True,
        context="function_args"
    )

    # Ensure result is a dict
    if not isinstance(result, dict):
        logging.warning(
            f"Function args parsed to non-dict type: {type(result)}. "
            f"Returning default."
        )
        return default

    return result


def _truncate(text: str, max_length: int = 200) -> str:
    """Truncate text for logging purposes."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _ctx(context: str) -> str:
    """Format context string for logging."""
    return f" [{context}]" if context else ""
