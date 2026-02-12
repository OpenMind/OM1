"""
Hot-reload manager for selective configuration updates.

This module provides selective hot-reload functionality for OM1 runtime,
allowing specific configuration fields to be updated without full system restart.
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class ReloadStrategy(Enum):
    """Strategy for handling configuration field changes."""

    HOT_RELOAD = "hot_reload"  # Can be reloaded without restart
    RESTART_REQUIRED = "restart"  # Requires full system restart
    VALIDATE_FIRST = "validate"  # Validate before applying


@dataclass
class FieldConfig:
    """Configuration metadata for a hot-reloadable field."""

    name: str
    strategy: ReloadStrategy
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""


@dataclass
class ConfigChange:
    """Represents a detected configuration change."""

    field_path: str  # e.g., "system_prompt_base" or "cortex_llm.config.temperature"
    old_value: Any
    new_value: Any
    strategy: ReloadStrategy


class HotReloadManager:
    """
    Manages selective hot-reload of configuration fields.

    This manager identifies which fields have changed, validates them,
    and applies hot-reloadable changes without full system restart.
    """

    def __init__(self):
        """Initialize the hot-reload manager."""
        self._lock = threading.Lock()
        self._field_configs: Dict[str, FieldConfig] = {}
        self._change_history: List[ConfigChange] = []
        self._max_history = 50

        # Register default hot-reloadable fields
        self._register_default_fields()

    def _register_default_fields(self):
        """Register default hot-reloadable fields for OM1."""
        # System prompts - safe to hot-reload
        self.register_field(
            "system_prompt_base",
            ReloadStrategy.HOT_RELOAD,
            validator=lambda v: isinstance(v, str) and len(v) > 0,
            description="Base system prompt for the agent",
        )

        self.register_field(
            "system_governance",
            ReloadStrategy.HOT_RELOAD,
            validator=lambda v: isinstance(v, str),
            description="Governance rules for the agent",
        )

        self.register_field(
            "system_prompt_examples",
            ReloadStrategy.HOT_RELOAD,
            validator=lambda v: isinstance(v, str),
            description="Example interactions for the agent",
        )

        # LLM parameters - safe to hot-reload
        # Note: These are nested in cortex_llm.config
        self.register_field(
            "cortex_llm.config.temperature",
            ReloadStrategy.VALIDATE_FIRST,
            validator=lambda v: isinstance(v, (int, float)) and 0 <= v <= 2,
            description="LLM temperature parameter",
        )

        self.register_field(
            "cortex_llm.config.top_p",
            ReloadStrategy.VALIDATE_FIRST,
            validator=lambda v: isinstance(v, (int, float)) and 0 <= v <= 1,
            description="LLM top-p sampling parameter",
        )

        self.register_field(
            "cortex_llm.config.max_tokens",
            ReloadStrategy.VALIDATE_FIRST,
            validator=lambda v: isinstance(v, int) and v > 0,
            description="Maximum tokens for LLM generation",
        )

        self.register_field(
            "cortex_llm.config.history_length",
            ReloadStrategy.VALIDATE_FIRST,
            validator=lambda v: isinstance(v, int) and v >= 0,
            description="Conversation history length",
        )

        # Fields that require restart
        self.register_field(
            "version", ReloadStrategy.RESTART_REQUIRED, description="Config version"
        )

        self.register_field(
            "hertz",
            ReloadStrategy.RESTART_REQUIRED,
            description="Execution frequency (tick rate)",
        )

        self.register_field(
            "name", ReloadStrategy.RESTART_REQUIRED, description="Agent name"
        )

        self.register_field(
            "api_key", ReloadStrategy.RESTART_REQUIRED, description="API key"
        )

        self.register_field(
            "robot_ip", ReloadStrategy.RESTART_REQUIRED, description="Robot IP address"
        )

        self.register_field(
            "agent_inputs",
            ReloadStrategy.RESTART_REQUIRED,
            description="Input orchestrator configuration",
        )

        self.register_field(
            "simulators",
            ReloadStrategy.RESTART_REQUIRED,
            description="Simulator configuration",
        )

        self.register_field(
            "agent_actions",
            ReloadStrategy.RESTART_REQUIRED,
            description="Action orchestrator configuration",
        )

        self.register_field(
            "cortex_llm.type",
            ReloadStrategy.RESTART_REQUIRED,
            description="LLM provider type",
        )

    def register_field(
        self,
        field_path: str,
        strategy: ReloadStrategy,
        validator: Optional[Callable[[Any], bool]] = None,
        description: str = "",
    ):
        """
        Register a field for hot-reload management.

        Parameters
        ----------
        field_path : str
            Dot-separated path to the field (e.g., "system_prompt_base")
        strategy : ReloadStrategy
            How to handle changes to this field
        validator : Callable, optional
            Function to validate the new value
        description : str, optional
            Human-readable description of the field
        """
        with self._lock:
            self._field_configs[field_path] = FieldConfig(
                name=field_path,
                strategy=strategy,
                validator=validator,
                description=description,
            )
            logging.debug(
                f"Registered field '{field_path}' with strategy: {strategy.value}"
            )

    def detect_changes(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> List[ConfigChange]:
        """
        Detect what configuration fields have changed.

        Parameters
        ----------
        old_config : dict
            Previous configuration
        new_config : dict
            New configuration

        Returns
        -------
        List[ConfigChange]
            List of detected changes
        """
        with self._lock:
            changes = []

            for field_path, field_config in self._field_configs.items():
                old_value = self._get_nested_value(old_config, field_path)
                new_value = self._get_nested_value(new_config, field_path)

                if old_value != new_value:
                    changes.append(
                        ConfigChange(
                            field_path=field_path,
                            old_value=old_value,
                            new_value=new_value,
                            strategy=field_config.strategy,
                        )
                    )

            return changes

    def validate_changes(self, changes: List[ConfigChange]) -> Dict[str, bool]:
        """
        Validate a list of configuration changes.

        Parameters
        ----------
        changes : List[ConfigChange]
            Changes to validate

        Returns
        -------
        Dict[str, bool]
            Map of field_path -> validation result
        """
        with self._lock:
            results = {}

            for change in changes:
                field_config = self._field_configs.get(change.field_path)

                if not field_config:
                    results[change.field_path] = False
                    logging.warning(f"Unknown field: {change.field_path}")
                    continue

                if field_config.validator:
                    try:
                        is_valid = field_config.validator(change.new_value)
                        results[change.field_path] = is_valid

                        if not is_valid:
                            logging.error(
                                f"Validation failed for '{change.field_path}': "
                                f"{change.new_value}"
                            )
                    except Exception as e:
                        logging.error(f"Validator error for '{change.field_path}': {e}")
                        results[change.field_path] = False
                else:
                    # No validator = assume valid
                    results[change.field_path] = True

            return results

    def categorize_changes(
        self, changes: List[ConfigChange]
    ) -> Dict[ReloadStrategy, List[ConfigChange]]:
        """
        Categorize changes by their reload strategy.

        Parameters
        ----------
        changes : List[ConfigChange]
            List of changes to categorize

        Returns
        -------
        Dict[ReloadStrategy, List[ConfigChange]]
            Changes grouped by strategy
        """
        with self._lock:
            categorized = {
                ReloadStrategy.HOT_RELOAD: [],
                ReloadStrategy.VALIDATE_FIRST: [],
                ReloadStrategy.RESTART_REQUIRED: [],
            }

            for change in changes:
                categorized[change.strategy].append(change)

            return categorized

    def get_hot_reloadable_fields(self) -> Set[str]:
        """
        Get set of all hot-reloadable field paths.

        Returns
        -------
        Set[str]
            Field paths that can be hot-reloaded
        """
        with self._lock:
            return {
                name
                for name, config in self._field_configs.items()
                if config.strategy
                in [ReloadStrategy.HOT_RELOAD, ReloadStrategy.VALIDATE_FIRST]
            }

    def get_restart_required_fields(self) -> Set[str]:
        """
        Get set of field paths that require restart.

        Returns
        -------
        Set[str]
            Field paths that require system restart
        """
        with self._lock:
            return {
                name
                for name, config in self._field_configs.items()
                if config.strategy == ReloadStrategy.RESTART_REQUIRED
            }

    def track_change(self, change: ConfigChange):
        """
        Add a change to the history.

        Parameters
        ----------
        change : ConfigChange
            Change to track
        """
        with self._lock:
            self._change_history.append(change)

            # Keep history limited
            if len(self._change_history) > self._max_history:
                self._change_history = self._change_history[-self._max_history :]

    def apply_changes(
        self, config: Dict[str, Any], changes: List[ConfigChange]
    ) -> Dict[str, bool]:
        """
        Apply validated hot-reloadable changes to configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary to modify in-place
        changes : List[ConfigChange]
            List of validated changes to apply

        Returns
        -------
        Dict[str, bool]
            Map of field_path -> success status
        """
        results = {}

        for change in changes:
            if change.strategy == ReloadStrategy.RESTART_REQUIRED:
                results[change.field_path] = False
                logging.warning(
                    f"Cannot hot-reload '{change.field_path}': restart required"
                )
                continue

            try:
                # Use lock only for individual operations
                with self._lock:
                    self._set_nested_value(config, change.field_path, change.new_value)

                # Track change separately (it has its own lock)
                self.track_change(change)

                results[change.field_path] = True
                logging.info(
                    f"Applied hot-reload: {change.field_path} = {change.new_value}"
                )
            except Exception as e:
                logging.error(
                    f"Failed to apply hot-reload for '{change.field_path}': {e}"
                )
                results[change.field_path] = False

        return results

    def get_change_history(self, limit: int = 10) -> List[ConfigChange]:
        """
        Get recent change history.

        Parameters
        ----------
        limit : int
            Maximum number of changes to return

        Returns
        -------
        List[ConfigChange]
            Recent configuration changes
        """
        with self._lock:
            return self._change_history[-limit:]

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        Get a value from nested dictionary using dot notation.

        Parameters
        ----------
        config : dict
            Configuration dictionary
        path : str
            Dot-separated path (e.g., "cortex_llm.config.temperature")

        Returns
        -------
        Any
            Value at the path, or None if not found
        """
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None

        return value

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """
        Set a value in nested dictionary using dot notation.

        Parameters
        ----------
        config : dict
            Configuration dictionary to modify
        path : str
            Dot-separated path
        value : Any
            Value to set
        """
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        logging.debug(f"Set nested value: {path} = {value}")
