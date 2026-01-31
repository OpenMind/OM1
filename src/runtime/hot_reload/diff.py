"""
Configuration Diff Engine

Provides deep comparison of configuration dictionaries to detect changes
at field level, including nested structures.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ConfigDiff:
    """
    Represents the differences between two configurations.
    
    Attributes:
        added: Fields that exist in new config but not in old
        removed: Fields that exist in old config but not in new
        modified: Fields that exist in both but have different values
        unchanged: Fields that are identical in both configs
    """
    added: Set[str] = field(default_factory=set)
    removed: Set[str] = field(default_factory=set)
    modified: Set[str] = field(default_factory=set)
    unchanged: Set[str] = field(default_factory=set)
    
    # Store old and new values for modified fields
    old_values: Dict[str, Any] = field(default_factory=dict)
    new_values: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.added or self.removed or self.modified)
    
    @property
    def changed_fields(self) -> Set[str]:
        """Get all fields that have changed (added, removed, or modified)."""
        return self.added | self.removed | self.modified
    
    def __repr__(self) -> str:
        if not self.has_changes:
            return "ConfigDiff(no changes)"
        
        parts = []
        if self.added:
            parts.append(f"added={self.added}")
        if self.removed:
            parts.append(f"removed={self.removed}")
        if self.modified:
            parts.append(f"modified={self.modified}")
        
        return f"ConfigDiff({', '.join(parts)})"


def _hash_value(value: Any) -> str:
    """
    Compute a hash for any JSON-serializable value.
    
    This is used for efficient comparison of complex nested structures.
    """
    serialized = json.dumps(value, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


def _deep_equals(value1: Any, value2: Any) -> bool:
    """
    Perform deep equality comparison of two values.
    
    Handles nested dicts, lists, and primitive types correctly.
    This fixes the bug in PR #1312 where only length/identity was checked.
    """
    # Same type check
    if type(value1) != type(value2):
        return False
    
    # Dict comparison
    if isinstance(value1, dict):
        if set(value1.keys()) != set(value2.keys()):
            return False
        return all(
            _deep_equals(value1[k], value2[k]) 
            for k in value1.keys()
        )
    
    # List comparison
    if isinstance(value1, list):
        if len(value1) != len(value2):
            return False
        return all(
            _deep_equals(v1, v2) 
            for v1, v2 in zip(value1, value2)
        )
    
    # Primitive comparison
    return value1 == value2


def compute_config_diff(
    old_config: Dict[str, Any],
    new_config: Dict[str, Any],
    use_hash_optimization: bool = True
) -> ConfigDiff:
    """
    Compute the differences between two configuration dictionaries.
    
    Args:
        old_config: The previous configuration
        new_config: The new configuration
        use_hash_optimization: Use hash-based comparison for complex values
            (faster for large nested structures)
    
    Returns:
        ConfigDiff object describing all changes
    """
    diff = ConfigDiff()
    
    old_keys = set(old_config.keys())
    new_keys = set(new_config.keys())
    
    # Find added fields
    diff.added = new_keys - old_keys
    for key in diff.added:
        diff.new_values[key] = new_config[key]
    
    # Find removed fields
    diff.removed = old_keys - new_keys
    for key in diff.removed:
        diff.old_values[key] = old_config[key]
    
    # Find modified and unchanged fields
    common_keys = old_keys & new_keys
    
    for key in common_keys:
        old_val = old_config[key]
        new_val = new_config[key]
        
        # Determine if values are equal
        if use_hash_optimization and isinstance(old_val, (dict, list)):
            # Use hash for complex structures (more efficient)
            is_equal = _hash_value(old_val) == _hash_value(new_val)
        else:
            # Use deep comparison
            is_equal = _deep_equals(old_val, new_val)
        
        if is_equal:
            diff.unchanged.add(key)
        else:
            diff.modified.add(key)
            diff.old_values[key] = old_val
            diff.new_values[key] = new_val
    
    return diff


def get_nested_diff(
    old_config: Dict[str, Any],
    new_config: Dict[str, Any],
    path_prefix: str = ""
) -> List[Tuple[str, str, Any, Any]]:
    """
    Get detailed nested diff showing exact path of changes.
    
    Returns list of (path, change_type, old_value, new_value) tuples.
    Useful for debugging and detailed logging.
    
    Args:
        old_config: The previous configuration
        new_config: The new configuration
        path_prefix: Current path prefix for nested calls
        
    Returns:
        List of change tuples
    """
    changes: List[Tuple[str, str, Any, Any]] = []
    
    old_keys = set(old_config.keys())
    new_keys = set(new_config.keys())
    
    # Added keys
    for key in (new_keys - old_keys):
        path = f"{path_prefix}.{key}" if path_prefix else key
        changes.append((path, "added", None, new_config[key]))
    
    # Removed keys
    for key in (old_keys - new_keys):
        path = f"{path_prefix}.{key}" if path_prefix else key
        changes.append((path, "removed", old_config[key], None))
    
    # Modified keys
    for key in (old_keys & new_keys):
        path = f"{path_prefix}.{key}" if path_prefix else key
        old_val = old_config[key]
        new_val = new_config[key]
        
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            # Recurse into nested dicts
            changes.extend(get_nested_diff(old_val, new_val, path))
        elif not _deep_equals(old_val, new_val):
            changes.append((path, "modified", old_val, new_val))
    
    return changes
