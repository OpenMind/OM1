import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
import typer

from cli import (
    _check_action_exists,
    _check_api_key,
    _check_background_exists,
    _check_class_in_dir,
    _check_input_exists,
    _check_llm_exists,
    _check_simulator_exists,
    _print_config_summary,
    _resolve_config_path,
    _validate_components,
    _validate_mode_components,
    list_configs,
    modes,
    validate_config,
)


def test_resolve_existing_absolute_path(tmp_path):
    """Test resolving an existing absolute path."""
    config_file = tmp_path / "test.json5"
    config_file.write_text("{}")
    result = _resolve_config_path(str(config_file))
    assert result == str(config_file.absolute())


def test_resolve_existing_path_with_extension(tmp_path):
    """Test resolving a path with .json5 extension."""
    config_file = tmp_path / "test.json5"
    config_file.write_text("{}")
    result = _resolve_config_path(str(tmp_path / "test"))
    assert result == str(config_file.absolute())


def test_resolve_config_in_config_dir(tmp_path):
    """Test resolving config from config directory."""
    with patch("os.path.dirname", return_value=str(tmp_path)):
        config_dir = tmp_path / ".." / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "test.json5"
        config_file.write_text("{}")

        result = _resolve_config_path("test")
        assert result.endswith("test.json5")


def test_resolve_nonexistent_config_raises_error():
    """Test that nonexistent config raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        _resolve_config_path("nonexistent_config_that_does_not_exist")
    assert "Configuration 'nonexistent_config_that_does_not_exist' not found" in str(
        exc_info.value
    )


def test_class_exists_in_directory(tmp_path):
    """Test finding a class in a directory."""
    test_file = tmp_path / "test_module.py"
    test_file.write_text("class TestClass:\n    pass\n")
    assert _check_class_in_dir(str(tmp_path), "TestClass") is True


def test_class_not_exists_in_directory(tmp_path):
    """Test when class doesn't exist in directory."""
    test_file = tmp_path / "test_module.py"
    test_file.write_text("class OtherClass:\n    pass\n")
    assert _check_class_in_dir(str(tmp_path), "TestClass") is False


def test_directory_not_exists():
    """Test when directory doesn't exist."""
    assert _check_class_in_dir("/nonexistent/directory/path", "TestClass") is False


def test_skip_init_file(tmp_path):
    """Test that __init__.py files are skipped."""
    init_file = tmp_path / "__init__.py"
    init_file.write_text("class TestClass:\n    pass\n")
    assert _check_class_in_dir(str(tmp_path), "TestClass") is False


def test_handles_syntax_errors_gracefully(tmp_path):
    """Test that syntax errors in files are handled gracefully."""
    test_file = tmp_path / "broken.py"
    test_file.write_text("class BrokenClass\n    # Missing colon\n")
    assert _check_class_in_dir(str(tmp_path), "BrokenClass") is False


def test_check_input_exists():
    """Test checking if input type exists."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = True
        assert _check_input_exists("TestInput") is True
        mock_check_class.assert_called_once()


def test_check_input_not_exists():
    """Test checking if input type doesn't exist."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = False
        assert _check_input_exists("NonexistentInput") is False


def test_check_llm_exists():
    """Test checking if LLM type exists."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = True
        assert _check_llm_exists("TestLLM") is True


def test_check_llm_not_exists():
    """Test checking if LLM type doesn't exist."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = False
        assert _check_llm_exists("NonexistentLLM") is False


def test_check_simulator_exists():
    """Test checking if simulator type exists."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = True
        assert _check_simulator_exists("TestSimulator") is True


def test_check_simulator_not_exists():
    """Test checking if simulator type doesn't exist."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = False
        assert _check_simulator_exists("NonexistentSimulator") is False


def test_check_action_exists():
    """Test checking if action exists."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        assert _check_action_exists("test_action") is True


def test_check_action_not_exists():
    """Test checking if action doesn't exist."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = False
        assert _check_action_exists("nonexistent_action") is False


def test_check_background_exists():
    """Test checking if background type exists."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = True
        assert _check_background_exists("TestBackground") is True


def test_check_background_not_exists():
    """Test checking if background type doesn't exist."""
    with patch("cli._check_class_in_dir") as mock_check_class:
        mock_check_class.return_value = False
        assert _check_background_exists("NonexistentBackground") is False


def test_check_api_key_no_key_warning(capsys):
    """Test warning when no API key is configured."""
    with patch.dict(os.environ, {}, clear=True):
        raw_config = {"api_key": ""}
        _check_api_key(raw_config, verbose=False)
        captured = capsys.readouterr()
        assert "Warning: No API key configured" in captured.out


def test_check_api_key_env_verbose(capsys):
    """Test API key from environment with verbose mode."""
    with patch.dict(os.environ, {"OM_API_KEY": "test_key"}, clear=True):
        raw_config = {"api_key": ""}
        _check_api_key(raw_config, verbose=True)
        captured = capsys.readouterr()
        assert "API key configured (from environment)" in captured.out


def test_check_api_key_config_verbose(capsys):
    """Test API key from config with verbose mode."""
    with patch.dict(os.environ, {}, clear=True):
        raw_config = {"api_key": "config_key"}
        _check_api_key(raw_config, verbose=True)
        captured = capsys.readouterr()
        assert "API key configured" in captured.out


def test_check_api_key_openmind_free_shows_warning(capsys):
    """Test warning with openmind_free key."""
    with patch.dict(os.environ, {}, clear=True):
        raw_config = {"api_key": "openmind_free"}
        _check_api_key(raw_config, verbose=False)
        captured = capsys.readouterr()
        assert "Warning: No API key configured" in captured.out


def test_print_config_summary_single_mode(capsys):
    """Test printing summary for single-mode config."""
    raw_config = {
        "name": "Test Config",
        "hertz": 10.0,
        "agent_inputs": [{"type": "input1"}],
        "agent_actions": [{"name": "action1"}],
    }
    _print_config_summary(raw_config, is_multi_mode=False)
    captured = capsys.readouterr()
    assert "Type: Single-mode" in captured.out
    assert "Test Config" in captured.out
    assert "10.0 Hz" in captured.out


def test_print_config_summary_multi_mode(capsys):
    """Test printing summary for multi-mode config."""
    raw_config = {
        "name": "Multi Mode Config",
        "default_mode": "mode1",
        "modes": {"mode1": {}, "mode2": {}},
        "transition_rules": [{"from": "mode1", "to": "mode2"}],
    }
    _print_config_summary(raw_config, is_multi_mode=True)
    captured = capsys.readouterr()
    assert "Type: Multi-mode" in captured.out
    assert "Multi Mode Config" in captured.out
    assert "Default Mode: mode1" in captured.out
    assert "Modes: 2" in captured.out


def test_validate_mode_components_valid():
    """Test validating mode with all valid components."""
    with (
        patch("cli._check_input_exists") as mock_input,
        patch("cli._check_llm_exists") as mock_llm,
        patch("cli._check_action_exists") as mock_action,
    ):

        mock_input.return_value = True
        mock_llm.return_value = True
        mock_action.return_value = True

        mode_data = {
            "agent_inputs": [{"type": "TestInput"}],
            "cortex_llm": {"type": "TestLLM"},
            "agent_actions": [{"name": "test_action"}],
        }

        errors, warnings = _validate_mode_components(
            "test_mode", mode_data, verbose=False
        )
        assert errors == []
        assert warnings == []


def test_validate_mode_components_missing_input():
    """Test that missing input creates error."""
    with patch("cli._check_input_exists") as mock_input:
        mock_input.return_value = False

        mode_data = {
            "agent_inputs": [{"type": "MissingInput"}],
        }

        errors, warnings = _validate_mode_components(
            "test_mode", mode_data, verbose=False, allow_missing=False
        )
        assert len(errors) == 1
        assert "MissingInput" in errors[0]


def test_validate_mode_components_missing_input_allowed():
    """Test that missing input creates warning when allowed."""
    with patch("cli._check_input_exists") as mock_input:
        mock_input.return_value = False

        mode_data = {
            "agent_inputs": [{"type": "MissingInput"}],
        }

        errors, warnings = _validate_mode_components(
            "test_mode", mode_data, verbose=False, allow_missing=True
        )
        assert len(errors) == 0
        assert len(warnings) == 1
        assert "MissingInput" in warnings[0]


def test_validate_mode_components_skip_inputs():
    """Test skipping input validation."""
    mode_data = {
        "agent_inputs": [{"type": "AnyInput"}],
    }

    errors, warnings = _validate_mode_components(
        "test_mode", mode_data, verbose=False, skip_inputs=True
    )
    assert errors == []
    assert warnings == []


def test_validate_mode_components_simulators():
    """Test simulator validation."""
    with patch("cli._check_simulator_exists") as mock_simulator:
        mock_simulator.return_value = True

        mode_data = {
            "simulators": [{"type": "TestSimulator"}],
        }

        errors, warnings = _validate_mode_components(
            "test_mode", mode_data, verbose=False
        )
        assert errors == []
        mock_simulator.assert_called_once_with("TestSimulator")


def test_validate_mode_components_backgrounds():
    """Test background validation."""
    with patch("cli._check_background_exists") as mock_background:
        mock_background.return_value = True

        mode_data = {
            "backgrounds": [{"type": "TestBackground"}],
        }

        errors, warnings = _validate_mode_components(
            "test_mode", mode_data, verbose=False
        )
        assert errors == []
        mock_background.assert_called_once_with("TestBackground")


def test_validate_components_single_mode():
    """Test validating single-mode configuration."""
    with patch("cli._validate_mode_components") as mock_validate_mode:
        mock_validate_mode.return_value = ([], [])

        raw_config = {
            "agent_inputs": [],
            "agent_actions": [],
        }

        _validate_components(raw_config, is_multi_mode=False, verbose=False)
        mock_validate_mode.assert_called_once()


def test_validate_components_multi_mode():
    """Test validating multi-mode configuration."""
    with (
        patch("cli._validate_mode_components") as mock_validate_mode,
        patch("cli._check_llm_exists") as mock_llm,
    ):

        mock_llm.return_value = True
        mock_validate_mode.return_value = ([], [])

        raw_config = {
            "cortex_llm": {"type": "GlobalLLM"},
            "modes": {
                "mode1": {"agent_inputs": []},
                "mode2": {"agent_inputs": []},
            },
        }

        _validate_components(raw_config, is_multi_mode=True, verbose=False)
        assert mock_validate_mode.call_count == 2


def test_validate_components_with_errors():
    """Test that errors raise ValueError."""
    with patch("cli._validate_mode_components") as mock_validate_mode:
        mock_validate_mode.return_value = (["Error 1", "Error 2"], [])

        raw_config = {"modes": {"mode1": {}}}

        with pytest.raises(ValueError, match="Component validation failed"):
            _validate_components(raw_config, is_multi_mode=True, verbose=False)


def test_validate_components_with_warnings(capsys):
    """Test that warnings are printed but no exception raised."""
    with patch("cli._validate_mode_components") as mock_validate_mode:
        mock_validate_mode.return_value = ([], ["Warning 1"])

        raw_config = {"modes": {"mode1": {}}}

        _validate_components(raw_config, is_multi_mode=True, verbose=False)
        captured = capsys.readouterr()
        assert "Warning 1" in captured.out


def test_list_configs_categorizes_correctly(capsys):
    """Test that configs are categorized correctly."""
    with (
        patch("os.listdir") as mock_listdir,
        patch("os.path.exists") as mock_exists,
        patch("builtins.open", new_callable=mock_open) as mock_file,
    ):

        mock_exists.return_value = True
        mock_listdir.return_value = ["mode_config.json5", "single_config.json5"]

        # Create a side effect to return different content
        def open_side_effect(path, mode):
            if "mode_config" in str(path):
                return mock_open(
                    read_data='{"modes": {}, "default_mode": "test"}'
                ).return_value
            else:
                return mock_open(read_data='{"name": "single"}').return_value

        mock_file.side_effect = open_side_effect

        list_configs()
        captured = capsys.readouterr()
        assert "Mode-Aware Configurations:" in captured.out
        assert "Standard Configurations:" in captured.out


def test_modes_displays_config_info(capsys):
    """Test that modes command displays configuration info."""
    with patch("cli.load_mode_config") as mock_load_config:
        mock_config = MagicMock()
        mock_config.name = "Test Config"
        mock_config.default_mode = "mode1"
        mock_config.allow_manual_switching = True
        mock_config.mode_memory_enabled = True
        mock_config.global_lifecycle_hooks = []
        mock_config.modes = {
            "mode1": MagicMock(
                display_name="Mode 1",
                description="Test mode",
                hertz=10.0,
                timeout_seconds=None,
                _raw_inputs=[],
                _raw_actions=[],
                lifecycle_hooks=[],
            )
        }
        mock_config.transition_rules = []
        mock_load_config.return_value = mock_config

        modes("test_config")
        captured = capsys.readouterr()
        assert "Test Config" in captured.out
        assert "Default Mode: mode1" in captured.out


def test_modes_handles_file_not_found():
    """Test that modes command handles FileNotFoundError."""
    with patch("cli.load_mode_config") as mock_load_config:
        mock_load_config.side_effect = FileNotFoundError()

        with pytest.raises(typer.Exit):
            modes("nonexistent_config")


def test_validate_config_success(capsys):
    """Test successful config validation."""
    with (
        patch("cli._resolve_config_path") as mock_resolve,
        patch("builtins.open", new_callable=mock_open, read_data='{"name": "test"}'),
        patch("cli.validate"),
        patch("cli._validate_components"),
        patch("cli._check_api_key"),
    ):

        mock_resolve.return_value = "/path/to/config.json5"

        validate_config("test", verbose=False, check_components=False)
        captured = capsys.readouterr()
        assert "Configuration is valid!" in captured.out


def test_validate_config_invalid_json():
    """Test validation with invalid JSON."""
    with (
        patch("cli._resolve_config_path") as mock_resolve,
        patch("builtins.open", new_callable=mock_open, read_data="invalid json{"),
    ):

        mock_resolve.return_value = "/path/to/config.json5"

        with pytest.raises(typer.Exit):
            validate_config("test")


def test_validate_config_file_not_found():
    """Test validation with missing file."""
    with patch("cli._resolve_config_path") as mock_resolve:
        mock_resolve.side_effect = FileNotFoundError()

        with pytest.raises(typer.Exit):
            validate_config("test")
