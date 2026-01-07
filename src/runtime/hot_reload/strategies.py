"""
Reload strategies for different configuration fields.

This module defines which configuration fields can be hot-reloaded
without restarting the system, and which require a full restart.
"""

import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set


class ReloadStrategy(Enum):
    """
    Defines how a configuration field should be reloaded.

    HOT_RELOAD: Field can be updated immediately without restart.
                Safe for runtime modification.

    VALIDATE_FIRST: Field requires validation before applying.
                    Will rollback if validation fails.

    RESTART_REQUIRED: Field change requires full system restart.
                      Cannot be applied at runtime safely.

    IGNORE: Field changes should be ignored (internal/computed fields).
    """

    HOT_RELOAD = "hot_reload"
    VALIDATE_FIRST = "validate_first"
    RESTART_REQUIRED = "restart_required"
    IGNORE = "ignore"


# Fields that can be hot-reloaded without system restart
HOT_RELOADABLE_FIELDS: Set[str] = {
    "system_prompt_base",
    "system_governance",
    "system_prompt_examples",
    "hertz",
    "name",
}

# Fields that require validation before applying
VALIDATE_FIRST_FIELDS: Set[str] = {
    "hertz",  # Must be positive number
}

# Fields that require full system restart
RESTART_REQUIRED_FIELDS: Set[str] = {
    "cortex_llm",
    "agent_inputs",
    "agent_actions",
    "simulators",
    "backgrounds",
    "api_key",
    "robot_ip",
    "URID",
    "unitree_ethernet",
    "version",
    "mode",
    "action_execution_mode",
    "action_dependencies",
}

# Fields to ignore (internal/computed)
IGNORE_FIELDS: Set[str] = {
    "_internal",
    "_computed",
}


def get_field_strategy(field_name: str) -> ReloadStrategy:
    """
    Determine the reload strategy for a given configuration field.

    Parameters
    ----------
    field_name : str
        The name of the configuration field.

    Returns
    -------
    ReloadStrategy
        The appropriate reload strategy for the field.
    """
    # Check for nested field (e.g., "cortex_llm.config.model")
    root_field = field_name.split(".")[0]

    if root_field in IGNORE_FIELDS:
        return ReloadStrategy.IGNORE

    if root_field in RESTART_REQUIRED_FIELDS:
        return ReloadStrategy.RESTART_REQUIRED

    if root_field in VALIDATE_FIRST_FIELDS:
        return ReloadStrategy.VALIDATE_FIRST

    if root_field in HOT_RELOADABLE_FIELDS:
        return ReloadStrategy.HOT_RELOAD

    # Default to restart required for unknown fields (safe default)
    logging.warning(
        f"Unknown config field '{field_name}', defaulting to RESTART_REQUIRED"
    )
    return ReloadStrategy.RESTART_REQUIRED


# Validators for VALIDATE_FIRST fields
FieldValidator = Callable[[Any, Any], bool]

_FIELD_VALIDATORS: Dict[str, FieldValidator] = {}


def register_validator(field_name: str) -> Callable[[FieldValidator], FieldValidator]:
    """
    Decorator to register a validator for a field.

    Parameters
    ----------
    field_name : str
        The name of the field to validate.

    Returns
    -------
    Callable
        Decorator function.
    """

    def decorator(func: FieldValidator) -> FieldValidator:
        _FIELD_VALIDATORS[field_name] = func
        return func

    return decorator


def get_validator(field_name: str) -> Optional[FieldValidator]:
    """
    Get the validator function for a field.

    Parameters
    ----------
    field_name : str
        The name of the field.

    Returns
    -------
    Optional[FieldValidator]
        The validator function, or None if no validator is registered.
    """
    return _FIELD_VALIDATORS.get(field_name)


def validate_field(field_name: str, old_value: Any, new_value: Any) -> bool:
    """
    Validate a field change.

    Parameters
    ----------
    field_name : str
        The name of the field.
    old_value : Any
        The current value.
    new_value : Any
        The proposed new value.

    Returns
    -------
    bool
        True if the change is valid, False otherwise.
    """
    validator = get_validator(field_name)
    if validator is None:
        return True

    try:
        return validator(old_value, new_value)
    except Exception as e:
        logging.error(f"Validation error for field '{field_name}': {e}")
        return False


# Built-in validators
@register_validator("hertz")
def validate_hertz(old_value: Any, new_value: Any) -> bool:
    """Validate hertz field - must be positive number."""
    if not isinstance(new_value, (int, float)):
        logging.error(f"hertz must be a number, got {type(new_value).__name__}")
        return False
    if new_value <= 0:
        logging.error(f"hertz must be positive, got {new_value}")
        return False
    if new_value > 100:
        logging.warning(f"hertz value {new_value} is unusually high")
    return True


def categorize_changes(
    changed_fields: Dict[str, tuple],
) -> Dict[ReloadStrategy, Dict[str, tuple]]:
    """
    Categorize changed fields by their reload strategy.

    Parameters
    ----------
    changed_fields : Dict[str, tuple]
        Dictionary mapping field names to (old_value, new_value) tuples.

    Returns
    -------
    Dict[ReloadStrategy, Dict[str, tuple]]
        Changes grouped by reload strategy.
    """
    categorized: Dict[ReloadStrategy, Dict[str, tuple]] = {
        ReloadStrategy.HOT_RELOAD: {},
        ReloadStrategy.VALIDATE_FIRST: {},
        ReloadStrategy.RESTART_REQUIRED: {},
        ReloadStrategy.IGNORE: {},
    }

    for field_name, (old_val, new_val) in changed_fields.items():
        strategy = get_field_strategy(field_name)
        categorized[strategy][field_name] = (old_val, new_val)

    return categorized


def requires_restart(changed_fields: Dict[str, tuple]) -> bool:
    """
    Check if any changed field requires a full restart.

    Parameters
    ----------
    changed_fields : Dict[str, tuple]
        Dictionary mapping field names to (old_value, new_value) tuples.

    Returns
    -------
    bool
        True if any field requires restart, False otherwise.
    """
    for field_name in changed_fields:
        if get_field_strategy(field_name) == ReloadStrategy.RESTART_REQUIRED:
            return True
    return False


def get_hot_reloadable_changes(changed_fields: Dict[str, tuple]) -> Dict[str, tuple]:
    """
    Filter changes to only include hot-reloadable fields.

    Parameters
    ----------
    changed_fields : Dict[str, tuple]
        Dictionary mapping field names to (old_value, new_value) tuples.

    Returns
    -------
    Dict[str, tuple]
        Only the changes that can be hot-reloaded.
    """
    result = {}
    for field_name, values in changed_fields.items():
        strategy = get_field_strategy(field_name)
        if strategy in (ReloadStrategy.HOT_RELOAD, ReloadStrategy.VALIDATE_FIRST):
            result[field_name] = values
    return result
