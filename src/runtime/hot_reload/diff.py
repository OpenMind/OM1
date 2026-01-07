"""
Configuration diff engine for detecting field-level changes.

This module provides functionality to compare two configuration dictionaries
and identify which fields have changed, been added, or been removed.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ChangeType(Enum):
    """Type of change detected in a configuration field."""

    MODIFIED = "modified"
    ADDED = "added"
    REMOVED = "removed"


@dataclass
class FieldChange:
    """
    Represents a single field change in the configuration.

    Attributes
    ----------
    field_path : str
        Dot-separated path to the field (e.g., "cortex_llm.config.model").
    change_type : ChangeType
        The type of change (modified, added, or removed).
    old_value : Any
        The previous value (None for added fields).
    new_value : Any
        The new value (None for removed fields).
    """

    field_path: str
    change_type: ChangeType
    old_value: Any = None
    new_value: Any = None

    def __repr__(self) -> str:
        """Return string representation of the field change."""
        if self.change_type == ChangeType.ADDED:
            return f"FieldChange(ADDED '{self.field_path}': {self.new_value})"
        elif self.change_type == ChangeType.REMOVED:
            return f"FieldChange(REMOVED '{self.field_path}': was {self.old_value})"
        else:
            return f"FieldChange(MODIFIED '{self.field_path}': {self.old_value} -> {self.new_value})"


@dataclass
class ConfigDiff:
    """
    Result of comparing two configurations.

    Attributes
    ----------
    changes : List[FieldChange]
        List of all detected changes.
    has_changes : bool
        True if any changes were detected.
    changed_fields : Dict[str, Tuple[Any, Any]]
        Dictionary mapping field paths to (old_value, new_value) tuples.
    added_fields : Set[str]
        Set of field paths that were added.
    removed_fields : Set[str]
        Set of field paths that were removed.
    modified_fields : Set[str]
        Set of field paths that were modified.
    """

    changes: List[FieldChange] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if any changes were detected."""
        return len(self.changes) > 0

    @property
    def changed_fields(self) -> Dict[str, Tuple[Any, Any]]:
        """Get dictionary of all changes as (old_value, new_value) tuples."""
        return {c.field_path: (c.old_value, c.new_value) for c in self.changes}

    @property
    def added_fields(self) -> Set[str]:
        """Get set of added field paths."""
        return {c.field_path for c in self.changes if c.change_type == ChangeType.ADDED}

    @property
    def removed_fields(self) -> Set[str]:
        """Get set of removed field paths."""
        return {
            c.field_path for c in self.changes if c.change_type == ChangeType.REMOVED
        }

    @property
    def modified_fields(self) -> Set[str]:
        """Get set of modified field paths."""
        return {
            c.field_path for c in self.changes if c.change_type == ChangeType.MODIFIED
        }

    def get_root_fields(self) -> Set[str]:
        """
        Get the set of root-level fields that have changes.

        Returns
        -------
        Set[str]
            Set of root field names (first part of dot-separated paths).
        """
        return {c.field_path.split(".")[0] for c in self.changes}

    def get_changes_for_field(self, field_name: str) -> List[FieldChange]:
        """
        Get all changes related to a specific root field.

        Parameters
        ----------
        field_name : str
            The root field name to filter by.

        Returns
        -------
        List[FieldChange]
            List of changes for that field and its nested fields.
        """
        return [
            c
            for c in self.changes
            if c.field_path == field_name or c.field_path.startswith(f"{field_name}.")
        ]

    def __repr__(self) -> str:
        """Return string representation of the config diff."""
        if not self.has_changes:
            return "ConfigDiff(no changes)"
        return f"ConfigDiff({len(self.changes)} changes: {list(self.changed_fields.keys())})"


def _is_primitive(value: Any) -> bool:
    """Check if a value is a primitive type (not dict or list)."""
    return not isinstance(value, (dict, list))


def _compare_values(old_val: Any, new_val: Any) -> bool:
    """
    Compare two values for equality.

    Handles special cases like comparing dicts and lists.
    """
    if type(old_val) is not type(new_val):
        return False

    if isinstance(old_val, dict):
        if set(old_val.keys()) != set(new_val.keys()):
            return False
        return all(_compare_values(old_val[k], new_val[k]) for k in old_val.keys())

    if isinstance(old_val, list):
        if len(old_val) != len(new_val):
            return False
        return all(_compare_values(o, n) for o, n in zip(old_val, new_val))

    return old_val == new_val


def _diff_recursive(
    old_config: Dict[str, Any],
    new_config: Dict[str, Any],
    path_prefix: str = "",
    changes: Optional[List[FieldChange]] = None,
    max_depth: int = 10,
) -> List[FieldChange]:
    """
    Recursively compare two configuration dictionaries.

    Parameters
    ----------
    old_config : Dict[str, Any]
        The original configuration.
    new_config : Dict[str, Any]
        The new configuration.
    path_prefix : str
        Current path prefix for nested fields.
    changes : Optional[List[FieldChange]]
        List to accumulate changes (created if None).
    max_depth : int
        Maximum recursion depth to prevent infinite loops.

    Returns
    -------
    List[FieldChange]
        List of detected field changes.
    """
    if changes is None:
        changes = []

    if max_depth <= 0:
        logging.warning(f"Max diff depth reached at path: {path_prefix}")
        return changes

    old_keys = set(old_config.keys())
    new_keys = set(new_config.keys())

    # Find removed fields
    for key in old_keys - new_keys:
        field_path = f"{path_prefix}{key}" if path_prefix else key
        changes.append(
            FieldChange(
                field_path=field_path,
                change_type=ChangeType.REMOVED,
                old_value=old_config[key],
                new_value=None,
            )
        )

    # Find added fields
    for key in new_keys - old_keys:
        field_path = f"{path_prefix}{key}" if path_prefix else key
        changes.append(
            FieldChange(
                field_path=field_path,
                change_type=ChangeType.ADDED,
                old_value=None,
                new_value=new_config[key],
            )
        )

    # Find modified fields
    for key in old_keys & new_keys:
        field_path = f"{path_prefix}{key}" if path_prefix else key
        old_val = old_config[key]
        new_val = new_config[key]

        # Both are dicts - recurse
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            _diff_recursive(old_val, new_val, f"{field_path}.", changes, max_depth - 1)
        # Values are different
        elif not _compare_values(old_val, new_val):
            changes.append(
                FieldChange(
                    field_path=field_path,
                    change_type=ChangeType.MODIFIED,
                    old_value=old_val,
                    new_value=new_val,
                )
            )

    return changes


def diff_configs(old_config: Dict[str, Any], new_config: Dict[str, Any]) -> ConfigDiff:
    """
    Compare two configuration dictionaries and return the differences.

    Parameters
    ----------
    old_config : Dict[str, Any]
        The original configuration.
    new_config : Dict[str, Any]
        The new configuration to compare against.

    Returns
    -------
    ConfigDiff
        Object containing all detected changes.

    Examples
    --------
    >>> old = {"name": "bot", "hertz": 1, "prompt": "Hello"}
    >>> new = {"name": "bot", "hertz": 2, "prompt": "Hi"}
    >>> diff = diff_configs(old, new)
    >>> diff.has_changes
    True
    >>> diff.modified_fields
    {'hertz', 'prompt'}
    """
    if old_config is None:
        old_config = {}
    if new_config is None:
        new_config = {}

    changes = _diff_recursive(old_config, new_config)
    return ConfigDiff(changes=changes)


def get_nested_value(config: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    """
    Get a value from a nested configuration using dot notation.

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration dictionary.
    path : str
        Dot-separated path to the value (e.g., "cortex_llm.config.model").

    Returns
    -------
    Tuple[bool, Any]
        (exists, value) tuple. exists is False if path doesn't exist.
    """
    parts = path.split(".")
    current = config

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]

    return True, current


def set_nested_value(config: Dict[str, Any], path: str, value: Any) -> bool:
    """
    Set a value in a nested configuration using dot notation.

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration dictionary to modify.
    path : str
        Dot-separated path to the value.
    value : Any
        The value to set.

    Returns
    -------
    bool
        True if successful, False if path is invalid.
    """
    parts = path.split(".")
    current = config

    # Navigate to parent
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
        if not isinstance(current, dict):
            return False

    # Set the value
    current[parts[-1]] = value
    return True


def apply_changes(config: Dict[str, Any], changes: List[FieldChange]) -> Dict[str, Any]:
    """
    Apply a list of changes to a configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        The configuration to modify (will be copied).
    changes : List[FieldChange]
        List of changes to apply.

    Returns
    -------
    Dict[str, Any]
        New configuration with changes applied.
    """
    import copy

    result = copy.deepcopy(config)

    for change in changes:
        if change.change_type == ChangeType.REMOVED:
            # Remove the field
            parts = change.field_path.split(".")
            current = result
            for part in parts[:-1]:
                if part in current:
                    current = current[part]
                else:
                    break
            else:
                if parts[-1] in current:
                    del current[parts[-1]]

        elif change.change_type in (ChangeType.ADDED, ChangeType.MODIFIED):
            set_nested_value(result, change.field_path, change.new_value)

    return result
