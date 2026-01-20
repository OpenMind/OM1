import logging
from unittest.mock import mock_open, patch

import pytest
from jsonschema import ValidationError

from runtime.config import _load_schema, validate_config_schema


def test_load_schema_success():
    """Test successful loading of a schema file."""
    schema = _load_schema("single_mode_schema.json")
    assert isinstance(schema, dict)
    assert len(schema) > 0


def test_load_multi_mode_schema_success():
    """Test successful loading of multi-mode schema file."""
    schema = _load_schema("multi_mode_schema.json")
    assert isinstance(schema, dict)
    assert len(schema) > 0


def test_load_schema_file_not_found():
    """Test that FileNotFoundError is raised for non-existent schema file."""
    with pytest.raises(FileNotFoundError) as exc_info:
        _load_schema("nonexistent_schema.json")
    assert "Schema file not found" in str(exc_info.value)


def test_load_schema_returns_valid_json():
    """Test that loaded schema is valid JSON structure."""
    schema = _load_schema("single_mode_schema.json")
    assert "$schema" in schema or "type" in schema or "properties" in schema


@patch("builtins.open", mock_open(read_data='{"type": "object", "properties": {}}'))
@patch("pathlib.Path.exists", return_value=True)
def test_load_schema_with_mock_file(mock_exists):
    """Test schema loading with mocked file."""
    schema = _load_schema("test_schema.json")
    assert schema == {"type": "object", "properties": {}}


@patch("pathlib.Path.exists", return_value=False)
def test_load_schema_path_does_not_exist(mock_exists):
    """Test that FileNotFoundError is raised when path doesn't exist."""
    with pytest.raises(FileNotFoundError):
        _load_schema("missing_schema.json")


def test_validate_single_mode_config_minimal():
    """Test validation of minimal valid single-mode configuration."""
    config = {
        "version": "v1.0.0",
        "hertz": 10.0,
        "name": "test_config",
        "api_key": "test_key",
        "system_prompt_base": "You are a helpful assistant.",
        "system_governance": "Be helpful and harmless.",
        "system_prompt_examples": "Example: Q: Hello A: Hi there!",
        "agent_inputs": [],
        "cortex_llm": {"type": "test_llm"},
        "agent_actions": [],
    }
    validate_config_schema(config)


def test_validate_multi_mode_config_minimal():
    """Test validation of minimal valid multi-mode configuration."""
    config = {
        "version": "v1.0.0",
        "default_mode": "mode1",
        "api_key": "test_key",
        "system_governance": "Be helpful and harmless.",
        "cortex_llm": {"type": "test_llm"},
        "modes": {
            "mode1": {
                "display_name": "Test Mode",
                "description": "A test mode",
                "system_prompt_base": "You are a helpful assistant.",
                "hertz": 10.0,
                "agent_inputs": [],
                "agent_actions": [],
            }
        },
    }
    validate_config_schema(config)


def test_validate_config_selects_single_mode_schema():
    """Test that single-mode schema is selected when 'modes' key is absent."""
    config = {
        "name": "test_config",
        "version": "v1.0.0",
        "actions": [],
        "inputs": [],
        "backgrounds": [],
    }
    with patch("runtime.config._load_schema") as mock_load:
        mock_load.return_value = {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }
        validate_config_schema(config)
    mock_load.assert_called_once_with("single_mode_schema.json")


def test_validate_config_selects_multi_mode_schema():
    """Test that multi-mode schema is selected when 'modes' key is present."""
    config = {"name": "test_multi_mode", "version": "v1.0.0", "modes": {}}
    with patch("runtime.config._load_schema") as mock_load:
        mock_load.return_value = {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }
        validate_config_schema(config)
    mock_load.assert_called_once_with("multi_mode_schema.json")


def test_validate_config_schema_file_not_found():
    """Test that FileNotFoundError is raised when schema file is missing."""
    config = {"name": "test"}
    with patch(
        "runtime.config._load_schema", side_effect=FileNotFoundError("Schema not found")
    ):
        with pytest.raises(FileNotFoundError):
            validate_config_schema(config)


def test_validate_config_invalid_schema_raises_validation_error():
    """Test that ValidationError is raised for invalid configuration."""
    config = {"invalid_field": "value"}
    with pytest.raises(ValidationError):
        validate_config_schema(config)


def test_validate_config_logs_validation_error_with_path(caplog):
    """Test that validation errors are logged with field path."""
    config = {
        "name": 123,  # Should be string
        "version": "v1.0.0",
        "actions": [],
        "inputs": [],
        "backgrounds": [],
    }
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValidationError):
            validate_config_schema(config)
    assert "Schema validation failed" in caplog.text


def test_validate_config_logs_file_not_found_error(caplog):
    """Test that FileNotFoundError is logged."""
    config = {"name": "test"}
    with patch(
        "runtime.config._load_schema",
        side_effect=FileNotFoundError("Schema file missing"),
    ):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(FileNotFoundError):
                validate_config_schema(config)
        assert "Schema file missing" in caplog.text


def test_validate_config_handles_nested_validation_error(caplog):
    """Test that nested validation errors are properly logged."""
    config = {
        "name": "test",
        "version": "v1.0.0",
        "actions": [{"invalid_nested": True}],  # Invalid action structure
        "inputs": [],
        "backgrounds": [],
    }
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValidationError):
            validate_config_schema(config)
    assert "Schema validation failed" in caplog.text


def test_validate_config_empty_dict():
    """Test validation with empty configuration dictionary."""
    config = {}
    with pytest.raises(ValidationError):
        validate_config_schema(config)


def test_validate_config_with_additional_properties():
    """Test validation behavior with additional properties."""
    config = {
        "name": "test_config",
        "version": "v1.0.0",
        "actions": [],
        "inputs": [],
        "backgrounds": [],
        "extra_field": "should_be_validated_by_schema",
    }
    try:
        validate_config_schema(config)
    except ValidationError:
        pass


def test_validate_config_with_complex_modes():
    """Test validation with complex multi-mode configuration."""
    config = {
        "name": "complex_multi_mode",
        "version": "v1.0.0",
        "modes": {
            "mode1": {"actions": [], "inputs": [], "backgrounds": []},
            "mode2": {"actions": [], "inputs": [], "backgrounds": []},
        },
    }
    try:
        validate_config_schema(config)
    except ValidationError:
        pass


@patch("runtime.config.validate")
def test_validate_config_calls_jsonschema_validate(mock_validate):
    """Test that jsonschema.validate is called with correct parameters."""
    config = {"name": "test", "modes": {}}
    schema = {"type": "object"}

    with patch("runtime.config._load_schema", return_value=schema):
        validate_config_schema(config)
    mock_validate.assert_called_once_with(instance=config, schema=schema)


def test_validate_config_error_message_at_root(caplog):
    """Test error logging when validation fails at root level."""
    config = "not_a_dict"

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValidationError):
            validate_config_schema(config)  # type: ignore
        assert "root" in caplog.text or "Schema validation failed" in caplog.text
