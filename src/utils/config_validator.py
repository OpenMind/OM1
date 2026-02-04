"""
OM1 Configuration Validator

This module provides validation for OM1 JSON5 configuration files.
It ensures that all required fields are present and validates data types
and value ranges for various configuration options.

Author: Your Name
Date: 2026-02-04
"""

import json
import json5
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    level: ValidationLevel
    field: str
    message: str
    line_number: Optional[int] = None

    def __str__(self):
        location = f" (line {self.line_number})" if self.line_number else ""
        return f"[{self.level.value.upper()}]{location} {self.field}: {self.message}"


class ConfigValidator:
    """
    Validator for OM1 configuration files.
    
    Validates JSON5 configuration files used by OM1 agents.
    """

    # Required fields for different config sections
    REQUIRED_FIELDS = {
        "root": ["name", "inputs", "brain", "actions"],
        "brain": ["llm"],
        "llm": ["provider", "model"],
        "input": ["type"],
        "action": ["type"],
    }

    # Valid values for specific fields
    VALID_VALUES = {
        "llm_providers": [
            "openai", "anthropic", "gemini", "deepseek", 
            "xai", "meta", "nearai", "openmind"
        ],
        "input_types": [
            "camera", "webcam", "lidar", "microphone", 
            "text", "sensor", "ros2", "zenoh", "websocket"
        ],
        "action_types": [
            "move", "speak", "gesture", "navigation",
            "ros2", "zenoh", "websocket", "display"
        ],
        "tts_providers": ["elevenlabs", "riva", "google"],
        "asr_providers": ["google", "whisper", "deepgram"],
    }

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize validator with config file path.
        
        Args:
            config_path: Path to JSON5 configuration file
        """
        self.config_path = Path(config_path)
        self.validation_results: List[ValidationResult] = []
        self.config: Optional[Dict[str, Any]] = None

    def load_config(self) -> bool:
        """
        Load and parse JSON5 configuration file.
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json5.load(f)
            return True
        except FileNotFoundError:
            self._add_error("file", f"Configuration file not found: {self.config_path}")
            return False
        except json5.JSON5DecodeError as e:
            self._add_error("file", f"Invalid JSON5 syntax: {str(e)}")
            return False
        except Exception as e:
            self._add_error("file", f"Error loading config: {str(e)}")
            return False

    def validate(self) -> bool:
        """
        Perform full validation of configuration.
        
        Returns:
            True if validation passed (no errors), False otherwise
        """
        if not self.load_config():
            return False

        # Run all validation checks
        self._validate_required_fields()
        self._validate_brain_config()
        self._validate_inputs()
        self._validate_actions()
        self._validate_api_keys()
        self._validate_network_config()

        # Check if any errors were found
        return not any(r.level == ValidationLevel.ERROR for r in self.validation_results)

    def _validate_required_fields(self):
        """Validate that all required root fields are present"""
        for field in self.REQUIRED_FIELDS["root"]:
            if field not in self.config:
                self._add_error(field, f"Required field '{field}' is missing")

    def _validate_brain_config(self):
        """Validate brain/LLM configuration"""
        if "brain" not in self.config:
            return

        brain = self.config["brain"]
        
        # Check required brain fields
        for field in self.REQUIRED_FIELDS["brain"]:
            if field not in brain:
                self._add_error(f"brain.{field}", f"Required field '{field}' is missing in brain config")
                return

        # Validate LLM configuration
        llm = brain.get("llm", {})
        
        for field in self.REQUIRED_FIELDS["llm"]:
            if field not in llm:
                self._add_error(f"brain.llm.{field}", f"Required field '{field}' is missing in LLM config")

        # Check provider is valid
        provider = llm.get("provider", "").lower()
        if provider and provider not in self.VALID_VALUES["llm_providers"]:
            self._add_warning(
                "brain.llm.provider",
                f"Unknown LLM provider '{provider}'. Valid providers: {', '.join(self.VALID_VALUES['llm_providers'])}"
            )

        # Validate temperature if present
        if "temperature" in llm:
            temp = llm["temperature"]
            if not isinstance(temp, (int, float)) or not (0 <= temp <= 2):
                self._add_error(
                    "brain.llm.temperature",
                    f"Temperature must be a number between 0 and 2, got: {temp}"
                )

        # Validate max_tokens if present
        if "max_tokens" in llm:
            max_tokens = llm["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                self._add_error(
                    "brain.llm.max_tokens",
                    f"max_tokens must be a positive integer, got: {max_tokens}"
                )

    def _validate_inputs(self):
        """Validate input configurations"""
        if "inputs" not in self.config:
            return

        inputs = self.config["inputs"]
        if not isinstance(inputs, list):
            self._add_error("inputs", "Inputs must be a list")
            return

        for idx, input_config in enumerate(inputs):
            if not isinstance(input_config, dict):
                self._add_error(f"inputs[{idx}]", "Each input must be an object")
                continue

            # Check required fields
            if "type" not in input_config:
                self._add_error(f"inputs[{idx}].type", "Input type is required")
                continue

            input_type = input_config["type"]
            if input_type not in self.VALID_VALUES["input_types"]:
                self._add_warning(
                    f"inputs[{idx}].type",
                    f"Unknown input type '{input_type}'. Valid types: {', '.join(self.VALID_VALUES['input_types'])}"
                )

            # Validate camera/webcam specific fields
            if input_type in ["camera", "webcam"]:
                if "index" in input_config:
                    index = input_config["index"]
                    if not isinstance(index, int) or index < 0:
                        self._add_error(
                            f"inputs[{idx}].index",
                            f"Camera index must be a non-negative integer, got: {index}"
                        )

    def _validate_actions(self):
        """Validate action configurations"""
        if "actions" not in self.config:
            return

        actions = self.config["actions"]
        if not isinstance(actions, list):
            self._add_error("actions", "Actions must be a list")
            return

        for idx, action_config in enumerate(actions):
            if not isinstance(action_config, dict):
                self._add_error(f"actions[{idx}]", "Each action must be an object")
                continue

            # Check required fields
            if "type" not in action_config:
                self._add_error(f"actions[{idx}].type", "Action type is required")
                continue

            action_type = action_config["type"]
            if action_type not in self.VALID_VALUES["action_types"]:
                self._add_warning(
                    f"actions[{idx}].type",
                    f"Unknown action type '{action_type}'. Valid types: {', '.join(self.VALID_VALUES['action_types'])}"
                )

            # Validate TTS configuration if present
            if action_type == "speak":
                self._validate_tts_config(action_config, idx)

    def _validate_tts_config(self, action_config: Dict[str, Any], idx: int):
        """Validate Text-to-Speech configuration"""
        if "tts" in action_config:
            tts = action_config["tts"]
            if not isinstance(tts, dict):
                self._add_error(f"actions[{idx}].tts", "TTS config must be an object")
                return

            provider = tts.get("provider", "").lower()
            if provider and provider not in self.VALID_VALUES["tts_providers"]:
                self._add_warning(
                    f"actions[{idx}].tts.provider",
                    f"Unknown TTS provider '{provider}'. Valid providers: {', '.join(self.VALID_VALUES['tts_providers'])}"
                )

    def _validate_api_keys(self):
        """Check for API key placeholders that need to be replaced"""
        if "brain" in self.config and "llm" in self.config["brain"]:
            llm = self.config["brain"]["llm"]
            api_key = llm.get("api_key", "")
            
            # Check for common placeholder values
            placeholders = ["your_api_key_here", "openmind_free", "REPLACE_ME", ""]
            if any(placeholder in str(api_key).lower() for placeholder in placeholders):
                self._add_warning(
                    "brain.llm.api_key",
                    "API key appears to be a placeholder. Please set a valid API key."
                )

    def _validate_network_config(self):
        """Validate network-related configurations"""
        # Check for ROS2/Zenoh configurations
        if "inputs" in self.config:
            for idx, input_config in enumerate(self.config["inputs"]):
                if input_config.get("type") in ["ros2", "zenoh"]:
                    if "topic" not in input_config:
                        self._add_warning(
                            f"inputs[{idx}].topic",
                            "Network inputs should specify a topic"
                        )

        if "actions" in self.config:
            for idx, action_config in enumerate(self.config["actions"]):
                if action_config.get("type") in ["ros2", "zenoh"]:
                    if "topic" not in action_config:
                        self._add_warning(
                            f"actions[{idx}].topic",
                            "Network actions should specify a topic"
                        )

    def _add_error(self, field: str, message: str):
        """Add an error to validation results"""
        self.validation_results.append(
            ValidationResult(ValidationLevel.ERROR, field, message)
        )

    def _add_warning(self, field: str, message: str):
        """Add a warning to validation results"""
        self.validation_results.append(
            ValidationResult(ValidationLevel.WARNING, field, message)
        )

    def _add_info(self, field: str, message: str):
        """Add an info message to validation results"""
        self.validation_results.append(
            ValidationResult(ValidationLevel.INFO, field, message)
        )

    def get_results(self) -> List[ValidationResult]:
        """Get all validation results"""
        return self.validation_results

    def print_results(self):
        """Print validation results in a formatted way"""
        if not self.validation_results:
            print(f"✅ Configuration file '{self.config_path}' is valid!")
            return

        # Group results by level
        errors = [r for r in self.validation_results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.validation_results if r.level == ValidationLevel.WARNING]
        infos = [r for r in self.validation_results if r.level == ValidationLevel.INFO]

        print(f"\n📋 Validation Results for '{self.config_path}':\n")

        if errors:
            print(f"❌ Errors ({len(errors)}):")
            for result in errors:
                print(f"  {result}")
            print()

        if warnings:
            print(f"⚠️  Warnings ({len(warnings)}):")
            for result in warnings:
                print(f"  {result}")
            print()

        if infos:
            print(f"ℹ️  Info ({len(infos)}):")
            for result in infos:
                print(f"  {result}")
            print()

        # Summary
        if errors:
            print(f"❌ Validation FAILED with {len(errors)} error(s)")
        else:
            print(f"✅ Validation PASSED (with {len(warnings)} warning(s))")


def validate_config_file(config_path: Union[str, Path]) -> bool:
    """
    Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if validation passed, False otherwise
    """
    validator = ConfigValidator(config_path)
    is_valid = validator.validate()
    validator.print_results()
    return is_valid


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python config_validator.py <config_file.json5>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    is_valid = validate_config_file(config_file)
    sys.exit(0 if is_valid else 1)
