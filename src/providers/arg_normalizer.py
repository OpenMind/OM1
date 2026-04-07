"""LLM argument normalization for scheduled cron job dispatch.

Builds a per-action normalizer that tolerates the common ways LLMs mis-produce
arguments before the args dict is handed to a dataclass constructor:

  1. Field aliasing   — maps common synonyms to canonical field names
                        (e.g. ``sentence`` → ``action`` for Speak).
  2. Single-field     — if the input type has exactly one required field and
     heuristic          the LLM sent exactly one unrecognised key, map it
                        regardless of name.
  3. Unknown pruning  — fields with no match are dropped with a warning rather
                        than causing a ``TypeError`` at instantiation.
  4. Type coercion    — Enum (case-insensitive), int, float, bool, str, and
                        Optional[T] are all handled.
"""

import dataclasses
import logging
import typing
from enum import Enum
from typing import Any, Callable, Dict, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field-alias table
# Maps LLM-produced synonyms → canonical dataclass field name.
# Only applied when the alias target actually exists in the input schema.
# ---------------------------------------------------------------------------

_FIELD_ALIASES: Dict[str, str] = {
    # speak / TTS actions
    "sentence": "action",
    "text": "action",
    "message": "action",
    "speech": "action",
    "content": "action",
    "words": "action",
    "phrase": "action",
    "utterance": "action",
    # face / emotion actions
    "emotion": "action",
    "expression": "action",
    "face": "action",
    "mood": "action",
    "feeling": "action",
    "state": "action",
    # move / motor actions
    "movement": "action",
    "motion": "action",
    "direction": "action",
    "gesture": "action",
    # generic fallbacks
    "command": "action",
    "value": "action",
    "param": "action",
    "input": "action",
}

# Fields that are common alias targets — used to detect when the LLM puts
# the function name in a field instead of the real content.
_ALIAS_TARGETS: Set[str] = set(_FIELD_ALIASES.values())


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _unwrap_optional(tp: Any) -> Tuple[Any, bool]:
    """Return ``(inner_type, is_optional)`` for ``Optional[T]``, else ``(tp, False)``."""
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _coerce_value(value: Any, expected_type: Any, field_name: str, label: str) -> Any:
    """Coerce *value* to *expected_type*, handling Optional, Enum, and primitives."""
    inner_type, is_optional = _unwrap_optional(expected_type)

    if value is None:
        if is_optional:
            return None
        logger.warning("normalize_args[%s]: field '%s' is None but not Optional", label, field_name)
        return value

    # Enum: try direct construction first, then case-insensitive name/value match
    if isinstance(inner_type, type) and issubclass(inner_type, Enum):
        if isinstance(value, inner_type):
            return value
        # Direct construction (handles exact-match values)
        try:
            return inner_type(value)
        except (ValueError, KeyError):
            pass
        # Case-insensitive fallback
        if isinstance(value, str):
            needle = value.strip().lower()
            for member in inner_type:
                if member.value.lower() == needle or member.name.lower() == needle:
                    return member
        raise ValueError(
            f"normalize_args[{label}]: cannot coerce {value!r} to {inner_type.__name__}; "
            f"valid values: {[m.value for m in inner_type]}"
        )

    if inner_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    if inner_type is int:
        return int(value)

    if inner_type is float:
        return float(value)

    if inner_type is str:
        return str(value).strip() if isinstance(value, str) else str(value)

    return value


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_arg_normalizer(
    input_type: type,
    input_type_hints: Dict[str, Any],
    llm_label: str,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a normalization function tailored to *input_type*.

    Parameters
    ----------
    input_type:
        The dataclass that will be instantiated from the normalized dict.
    input_type_hints:
        ``typing.get_type_hints(input_type)`` — pre-computed by the caller.
    llm_label:
        The action's LLM label (used only in log messages).

    Returns
    -------
    A callable ``normalize(raw_args) -> clean_args`` that is safe to call
    from any thread.
    """
    # Compute required vs optional fields from dataclass metadata
    required: Set[str] = set()
    if dataclasses.is_dataclass(input_type):
        for f in dataclasses.fields(input_type):
            if (
                f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
            ):
                required.add(f.name)

    known_fields: Set[str] = set(input_type_hints.keys())

    def normalize(raw_args: Dict[str, Any]) -> Dict[str, Any]:
        # Guard: must be a dict
        if not isinstance(raw_args, dict):
            logger.warning(
                "normalize_args[%s]: expected dict, got %s — wrapping as {'action': value}",
                llm_label,
                type(raw_args).__name__,
            )
            raw_args = {"action": raw_args}

        result: Dict[str, Any] = {}
        unmatched_input: list = []  # [(orig_key, value)]
        unmatched_required = required.copy()

        for key, value in raw_args.items():
            if key in known_fields:
                # Heuristic: if this field is a common alias target (e.g. "action")
                # and its value equals the function label, the LLM confused the
                # function name with the field content.  Skip it so that alias keys
                # (e.g. "text", "sentence") can fill the field with the real value.
                if key in _ALIAS_TARGETS and value == llm_label:
                    logger.debug(
                        "normalize_args[%s]: field '%s' value equals label — "
                        "skipping to allow alias override",
                        llm_label,
                        key,
                    )
                    continue
                # Exact match
                result[key] = value
                unmatched_required.discard(key)
            elif key in _FIELD_ALIASES and _FIELD_ALIASES[key] in known_fields:
                # Alias match
                canonical = _FIELD_ALIASES[key]
                if canonical not in result:  # don't overwrite an already-matched field
                    logger.debug(
                        "normalize_args[%s]: aliasing '%s' → '%s'",
                        llm_label,
                        key,
                        canonical,
                    )
                    result[canonical] = value
                    unmatched_required.discard(canonical)
                else:
                    logger.debug(
                        "normalize_args[%s]: alias '%s' → '%s' skipped (field already set)",
                        llm_label,
                        key,
                        canonical,
                    )
            else:
                unmatched_input.append((key, value))

        # Single-field heuristic: one unmatched required field + one unmatched input key
        if len(unmatched_required) == 1 and len(unmatched_input) == 1:
            canonical = next(iter(unmatched_required))
            orig_key, value = unmatched_input[0]
            logger.info(
                "normalize_args[%s]: single-field heuristic '%s' → '%s'",
                llm_label,
                orig_key,
                canonical,
            )
            result[canonical] = value
            unmatched_required.discard(canonical)
            unmatched_input = []

        # Warn about dropped input keys
        for key, _ in unmatched_input:
            logger.warning(
                "normalize_args[%s]: unknown field '%s' not in schema — dropping",
                llm_label,
                key,
            )

        # Warn about still-missing required fields
        for field_name in unmatched_required:
            logger.warning(
                "normalize_args[%s]: required field '%s' missing from LLM output",
                llm_label,
                field_name,
            )

        # Type coercion pass
        coerced: Dict[str, Any] = {}
        for key, value in result.items():
            if key not in input_type_hints:
                continue
            try:
                coerced[key] = _coerce_value(value, input_type_hints[key], key, llm_label)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "normalize_args[%s]: coercion failed for '%s'=%r — keeping raw value. %s",
                    llm_label,
                    key,
                    value,
                    exc,
                )
                coerced[key] = value

        return coerced

    return normalize
