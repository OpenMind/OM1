"""Advanced configuration validation and management system."""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import json5
import jsonschema
from jsonschema import Draft7Validator, ValidationError


class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """A configuration validation issue."""
    severity: ValidationSeverity
    message: str
    path: str
    value: Any = None
    expected: Any = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    errors: List[ValidationIssue] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


class ConfigValidator:
    """Advanced configuration validator."""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, List[Callable]] = {}
        self._logger = logging.getLogger("config_validator")
        self._load_default_schemas()
    
    def _load_default_schemas(self):
        """Load default configuration schemas."""
        # Runtime configuration schema
        self.schemas["runtime"] = {
            "type": "object",
            "required": ["hertz", "cortex_llm", "agent_inputs", "agent_actions"],
            "properties": {
                "hertz": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 100.0,
                    "description": "Runtime frequency in Hz"
                },
                "cortex_llm": {
                    "type": "object",
                    "required": ["type", "config"],
                    "properties": {
                        "type": {"type": "string"},
                        "config": {"type": "object"}
                    }
                },
                "agent_inputs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "config"],
                        "properties": {
                            "type": {"type": "string"},
                            "config": {"type": "object"}
                        }
                    }
                },
                "agent_actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "config"],
                        "properties": {
                            "type": {"type": "string"},
                            "config": {"type": "object"}
                        }
                    }
                },
                "simulators": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "config"],
                        "properties": {
                            "type": {"type": "string"},
                            "config": {"type": "object"}
                        }
                    }
                },
                "backgrounds": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "config"],
                        "properties": {
                            "type": {"type": "string"},
                            "config": {"type": "object"}
                        }
                    }
                }
            }
        }
        
        # Mode configuration schema
        self.schemas["mode"] = {
            "type": "object",
            "required": ["name", "default_mode", "modes"],
            "properties": {
                "name": {"type": "string"},
                "default_mode": {"type": "string"},
                "allow_manual_switching": {"type": "boolean"},
                "mode_memory_enabled": {"type": "boolean"},
                "modes": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {
                            "type": "object",
                            "required": ["system_prompt_base", "hertz"],
                            "properties": {
                                "display_name": {"type": "string"},
                                "description": {"type": "string"},
                                "system_prompt_base": {"type": "string"},
                                "hertz": {
                                    "type": "number",
                                    "minimum": 0.1,
                                    "maximum": 100.0
                                },
                                "timeout_seconds": {"type": "number", "minimum": 1},
                                "remember_locations": {"type": "boolean"},
                                "save_interactions": {"type": "boolean"}
                            }
                        }
                    }
                }
            }
        }
    
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """Register a custom schema."""
        self.schemas[name] = schema
        self._logger.info(f"Registered schema: {name}")
    
    def register_custom_validator(self, config_type: str, validator: Callable[[Dict[str, Any]], ValidationResult]):
        """Register a custom validator for a configuration type."""
        if config_type not in self.custom_validators:
            self.custom_validators[config_type] = []
        self.custom_validators[config_type].append(validator)
        self._logger.info(f"Registered custom validator for: {config_type}")
    
    def validate(self, config: Dict[str, Any], config_type: str = "runtime") -> ValidationResult:
        """
        Validate a configuration against its schema.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration to validate
        config_type : str
            Type of configuration (runtime, mode, etc.)
            
        Returns
        -------
        ValidationResult
            Validation result with any issues found
        """
        result = ValidationResult(is_valid=True)
        
        # JSON Schema validation
        if config_type in self.schemas:
            schema = self.schemas[config_type]
            validator = Draft7Validator(schema)
            
            try:
                validator.validate(config)
            except ValidationError as e:
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Schema validation failed: {e.message}",
                    path=".".join(str(p) for p in e.absolute_path),
                    value=e.instance,
                    expected=e.validator_value
                )
                result.add_issue(issue)
        
        # Custom validators
        if config_type in self.custom_validators:
            for validator in self.custom_validators[config_type]:
                try:
                    custom_result = validator(config)
                    for issue in custom_result.issues:
                        result.add_issue(issue)
                except Exception as e:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Custom validator failed: {str(e)}",
                        path="custom_validator"
                    )
                    result.add_issue(issue)
        
        # Additional validation rules
        self._validate_runtime_specific(config, result)
        
        return result
    
    def _validate_runtime_specific(self, config: Dict[str, Any], result: ValidationResult):
        """Validate runtime-specific configuration rules."""
        # Check hertz value
        if "hertz" in config:
            hertz = config["hertz"]
            if hertz < 0.1:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Very low hertz value may cause performance issues",
                    path="hertz",
                    value=hertz
                ))
            elif hertz > 50:
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Very high hertz value may cause system overload",
                    path="hertz",
                    value=hertz
                ))
        
        # Check for required API keys
        if "api_key" in config:
            api_key = config["api_key"]
            if not api_key or api_key == "openmind_free":
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Using default API key, consider setting a real one",
                    path="api_key"
                ))
        
        # Check robot IP
        if "robot_ip" in config:
            robot_ip = config["robot_ip"]
            if not robot_ip or robot_ip == "192.168.0.241":
                result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Using default robot IP, verify this is correct",
                    path="robot_ip",
                    value=robot_ip
                ))
        
        # Validate LLM configuration
        if "cortex_llm" in config:
            llm_config = config["cortex_llm"]
            if "config" in llm_config:
                llm_inner_config = llm_config["config"]
                if "api_key" in llm_inner_config:
                    api_key = llm_inner_config["api_key"]
                    if not api_key or api_key == "openmind_free":
                        result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message="LLM using default API key",
                            path="cortex_llm.config.api_key"
                        ))


class ConfigWatcher:
    """File system watcher for configuration hot-reloading."""
    
    def __init__(self, config_path: str, callback: Callable[[str], None]):
        self.config_path = Path(config_path)
        self.callback = callback
        self._logger = logging.getLogger("config_watcher")
        self._last_modified = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start watching for configuration changes."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        self._logger.info(f"Started watching config: {self.config_path}")
    
    async def stop(self):
        """Stop watching for configuration changes."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._logger.info("Stopped watching config")
    
    async def _watch_loop(self):
        """Main watching loop."""
        while self._running:
            try:
                if self.config_path.exists():
                    current_modified = self.config_path.stat().st_mtime
                    if current_modified > self._last_modified:
                        self._last_modified = current_modified
                        self._logger.info(f"Config file changed: {self.config_path}")
                        self.callback(str(self.config_path))
                
                await asyncio.sleep(1.0)  # Check every second
            except Exception as e:
                self._logger.error(f"Error watching config: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error


class ConfigManager:
    """Advanced configuration manager with validation and hot-reloading."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.validator = ConfigValidator()
        self.watchers: Dict[str, ConfigWatcher] = {}
        self._logger = logging.getLogger("config_manager")
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    def load_config(self, config_name: str, validate: bool = True) -> Dict[str, Any]:
        """
        Load and validate a configuration file.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration file (without extension)
        validate : bool
            Whether to validate the configuration
            
        Returns
        -------
        Dict[str, Any]
            Loaded configuration
            
        Raises
        ------
        FileNotFoundError
            If configuration file doesn't exist
        ValidationError
            If validation fails and validate=True
        """
        config_path = self.config_dir / f"{config_name}.json5"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json5.load(f)
        
        if validate:
            # Determine config type based on content
            config_type = "mode" if "modes" in config else "runtime"
            result = self.validator.validate(config, config_type)
            
            if result.has_errors():
                error_messages = [f"{issue.path}: {issue.message}" for issue in result.errors]
                raise ValidationError(f"Configuration validation failed:\n" + "\n".join(error_messages))
            
            if result.has_warnings():
                warning_messages = [f"{issue.path}: {issue.message}" for issue in result.warnings]
                self._logger.warning(f"Configuration warnings:\n" + "\n".join(warning_messages))
        
        return config
    
    def save_config(self, config_name: str, config: Dict[str, Any], validate: bool = True):
        """
        Save a configuration file.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration file
        config : Dict[str, Any]
            Configuration to save
        validate : bool
            Whether to validate before saving
        """
        if validate:
            config_type = "mode" if "modes" in config else "runtime"
            result = self.validator.validate(config, config_type)
            if result.has_errors():
                error_messages = [f"{issue.path}: {issue.message}" for issue in result.errors]
                raise ValidationError(f"Configuration validation failed:\n" + "\n".join(error_messages))
        
        config_path = self.config_dir / f"{config_name}.json5"
        
        with open(config_path, 'w') as f:
            json5.dump(config, f, indent=2)
        
        self._logger.info(f"Saved configuration: {config_path}")
    
    def watch_config(self, config_name: str, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Start watching a configuration file for changes.
        
        Parameters
        ----------
        config_name : str
            Name of the configuration file to watch
        callback : Callable
            Callback function to call when config changes
        """
        config_path = self.config_dir / f"{config_name}.json5"
        
        def on_change(file_path: str):
            try:
                config = self.load_config(config_name, validate=True)
                callback(file_path, config)
            except Exception as e:
                self._logger.error(f"Error reloading config {config_name}: {e}")
        
        watcher = ConfigWatcher(str(config_path), on_change)
        self.watchers[config_name] = watcher
        asyncio.create_task(watcher.start())
    
    def stop_watching(self, config_name: str):
        """Stop watching a configuration file."""
        if config_name in self.watchers:
            asyncio.create_task(self.watchers[config_name].stop())
            del self.watchers[config_name]
    
    def stop_all_watchers(self):
        """Stop all configuration watchers."""
        for watcher in self.watchers.values():
            asyncio.create_task(watcher.stop())
        self.watchers.clear()


# Global configuration manager instance
config_manager = ConfigManager()
