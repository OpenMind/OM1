"""
Unit tests for the CLI module (src/cli.py).
Tests the Typer commands and helper functions directly.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Setup path *before* importing from src
current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.cli import (  # noqa: E402 (Import after path setup)
    _check_action_exists,
    _check_api_key,
    _check_input_exists,
    _check_llm_exists,
    _print_config_summary,
    _resolve_config_path,
    _validate_components,
    list_configs,
    modes,
    validate_config,
)


class TestCLICommands:

    def test_modes_command_success(self):
        config_name = "test_multi"

        mock_mode_config = MagicMock()
        mock_mode_config.name = "Test Multi Config"
        mock_mode_config.default_mode = "mode1"
        mock_mode_config.allow_manual_switching = True
        mock_mode_config.mode_memory_enabled = False
        mock_mode_config.global_lifecycle_hooks = ["hook1"]
        mock_mode_config.modes = {
            "mode1": MagicMock(
                display_name="Mode One",
                description="First mode",
                hertz=10.0,
                timeout_seconds=30,
                _raw_inputs=["inp1"],
                _raw_actions=["act1"],
                lifecycle_hooks=["lh1"],
            ),
        }
        mock_mode_config.transition_rules = [
            MagicMock(
                from_mode="*",
                to_mode="mode1",
                transition_type=MagicMock(value="keyword"),
                trigger_keywords=["hello"],
                priority=1,
                cooldown_seconds=5,
            )
        ]

        with patch(
            "src.cli.load_mode_config", return_value=mock_mode_config
        ) as mock_load_func:
            captured_output = []

            def mock_print(*args, **kwargs):
                captured_output.append(" ".join(map(str, args)))

            with patch("builtins.print", side_effect=mock_print):
                modes(config_name)

            mock_load_func.assert_called_once_with(config_name)

    def test_modes_command_file_not_found(self):
        config_name = "nonexistent"

        with patch("src.cli.load_mode_config", side_effect=FileNotFoundError()):
            with patch("logging.error") as mock_log_error:
                with patch("typer.Exit") as MockExit:
                    MockExit.side_effect = SystemExit(1)

                    with pytest.raises(SystemExit):
                        modes(config_name)

                    mock_log_error.assert_called_once_with(
                        f"Configuration file not found: {config_name}.json5"
                    )
                    MockExit.assert_called_once_with(1)

    def test_list_configs_directory_not_found(self):
        with patch("os.path.exists", return_value=False):
            captured_output = []

            def mock_print(*args, **kwargs):
                captured_output.append(" ".join(map(str, args)))

            with patch("builtins.print", side_effect=mock_print):
                list_configs()

            assert "Configuration directory not found" in captured_output

    def test_list_configs_success(self):
        config_dir = "/fake/config/dir"
        with patch("os.path.join", return_value=config_dir):
            with patch("os.path.exists", side_effect=lambda x: x == config_dir):
                with patch("os.listdir", return_value=["test.json5"]):
                    with patch(
                        "builtins.open",
                        mock_open(
                            read_data='{"name": "Test Config", "modes": {}, "default_mode": "default"}'
                        ),
                    ):
                        with patch(
                            "src.cli.json5.load",
                            return_value={
                                "name": "Test Config",
                                "modes": {},
                                "default_mode": "default",
                            },
                        ):
                            captured_output = []

                            def mock_print(*args, **kwargs):
                                captured_output.append(" ".join(map(str, args)))

                            with patch("builtins.print", side_effect=mock_print):
                                list_configs()

                            # Should contain mode-aware config
                            assert any(
                                "Mode-Aware Configurations:" in out
                                for out in captured_output
                            )

    def test_validate_config_success(self):
        config_name = "valid_config"
        config_path = f"/fake/path/{config_name}.json5"

        with patch("src.cli._resolve_config_path", return_value=config_path):
            with patch(
                "builtins.open", mock_open(read_data='{"name": "Valid", "hertz": 10}')
            ):
                with patch(
                    "src.cli.json5.load", return_value={"name": "Valid", "hertz": 10}
                ):
                    with patch("os.path.join") as mock_join:
                        schema_path = "/fake/schema/single_mode_schema.json"
                        mock_join.return_value = schema_path
                        with patch("builtins.open", mock_open(read_data="{}")):
                            with patch("json.load", return_value={}):
                                with patch("src.cli.validate") as mock_validate:
                                    with patch("builtins.print") as mock_print:
                                        validate_config(
                                            config_name,
                                            verbose=False,
                                            check_components=False,
                                        )

                                        # Verify core calls
                                        mock_validate.assert_called_once()
                                        # Verify success message was printed
                                        success_calls = [
                                            call
                                            for call in mock_print.call_args_list
                                            if "Configuration is valid!" in str(call)
                                        ]
                                        assert len(success_calls) > 0

    def test_validate_config_file_not_found(self):
        config_name = "invalid_config"
        with patch("src.cli._resolve_config_path", side_effect=FileNotFoundError()):
            with patch("typer.Exit") as MockExit:
                MockExit.side_effect = SystemExit(1)
                with pytest.raises(SystemExit):
                    validate_config(config_name, verbose=False, check_components=False)

                MockExit.assert_called_once_with(1)

    def test_validate_config_invalid_json5(self):
        config_name = "invalid_json"
        config_path = f"/fake/path/{config_name}.json5"

        with patch("src.cli._resolve_config_path", return_value=config_path):
            with patch("builtins.open", mock_open(read_data="{ invalid json }")):
                with patch(
                    "src.cli.json5.load", side_effect=ValueError("Invalid JSON")
                ):
                    with patch("typer.Exit") as MockExit:
                        MockExit.side_effect = SystemExit(1)
                        with pytest.raises(SystemExit):
                            validate_config(
                                config_name, verbose=False, check_components=False
                            )

                        MockExit.assert_called_once_with(1)

    def test_validate_config_schema_validation_error(self):
        config_name = "schema_fail"
        config_path = f"/fake/path/{config_name}.json5"

        with patch("src.cli._resolve_config_path", return_value=config_path):
            with patch("builtins.open", mock_open(read_data='{"bad_field": true}')):
                with patch("src.cli.json5.load", return_value={"bad_field": True}):
                    with patch("os.path.join") as mock_join:
                        schema_path = "/fake/schema/single_mode_schema.json"
                        mock_join.return_value = schema_path
                        with patch("builtins.open", mock_open(read_data="{}")):
                            with patch("json.load", return_value={}):
                                from jsonschema import ValidationError

                                with patch(
                                    "src.cli.validate",
                                    side_effect=ValidationError(
                                        "Required field missing"
                                    ),
                                ):
                                    with patch("typer.Exit") as MockExit:
                                        MockExit.side_effect = SystemExit(1)
                                        with pytest.raises(SystemExit):
                                            validate_config(
                                                config_name,
                                                verbose=True,
                                                check_components=False,
                                            )
                                        MockExit.assert_called_once_with(1)


class TestCLIFunctions:

    def test_resolve_config_path_by_name(self):
        config_name = "test_config"
        cli_file_dir = "/path/to/src"
        config_dir = "/path/to/config"
        config_path_json5 = f"{config_dir}/{config_name}.json5"
        expected_abs_path = f"/abs/{config_path_json5}"

        with patch("os.path.dirname", return_value=cli_file_dir):
            with patch(
                "os.path.join",
                side_effect=lambda x, y: (
                    config_dir if x == cli_file_dir and y == "../config" else f"{x}/{y}"
                ),
            ):
                with patch(
                    "os.path.exists", side_effect=lambda path: path == config_path_json5
                ):
                    with patch(
                        "os.path.abspath",
                        side_effect=lambda x: (
                            expected_abs_path if x == config_path_json5 else x
                        ),
                    ):
                        resolved = _resolve_config_path(config_name)
                        assert resolved == expected_abs_path

    def test_resolve_config_path_by_path(self):
        config_path = "/some/absolute/path/test.json5"

        with patch("os.path.exists", side_effect=lambda x: x == config_path):
            with patch("os.path.abspath", side_effect=lambda x: x):
                resolved = _resolve_config_path(config_path)
                assert resolved == config_path

    def test_resolve_config_path_not_found(self):
        config_name = "missing_config"
        cli_file_dir = "/path/to/src"
        config_dir = "/path/to/config"
        path1, path2, path3, path4 = (
            config_name,
            config_name + ".json5",
            f"{config_dir}/{config_name}",
            f"{config_dir}/{config_name}.json5",
        )

        with patch("os.path.dirname", return_value=cli_file_dir):
            with patch(
                "os.path.join",
                side_effect=lambda x, y: (
                    config_dir if x == cli_file_dir and y == "../config" else f"{x}/{y}"
                ),
            ):
                with patch("os.path.exists", return_value=False):
                    with pytest.raises(FileNotFoundError) as exc_info:
                        _resolve_config_path(config_name)
                    err_msg = str(exc_info.value)
                    for p in [path1, path2, path3, path4]:
                        assert p in err_msg

    def test_check_input_exists_true(self):
        input_type = "MockInput"
        plugin_dir = "/fake/src/inputs/plugins"
        plugin_filename = f"{input_type.lower()}.py"
        mock_file_content = f"class {input_type}:\n    pass\n"
        parsed_ast = ast.parse(mock_file_content)

        with patch("os.path.dirname", return_value="/fake/src"):
            with patch("os.path.join", return_value=plugin_dir):
                with patch("os.path.exists", return_value=True):
                    with patch("os.listdir", return_value=[plugin_filename]):
                        with patch(
                            "builtins.open", mock_open(read_data=mock_file_content)
                        ):
                            with patch("ast.parse", return_value=parsed_ast):
                                result = _check_input_exists(input_type)
                                assert result is True

    def test_check_input_exists_false(self):
        input_type = "NonExistentInput"
        plugin_dir = "/fake/src/inputs/plugins"

        with patch("os.path.dirname", return_value="/fake/src"):
            with patch("os.path.join", return_value=plugin_dir):
                with patch("os.path.exists", return_value=True):
                    with patch("os.listdir", return_value=[]):
                        result = _check_input_exists(input_type)
                        assert result is False

    def test_check_llm_exists_true(self):
        llm_type = "MockLLM"
        plugin_dir = "/fake/src/llm/plugins"
        plugin_filename = f"{llm_type.lower()}.py"
        mock_file_content = f"class {llm_type}:\n    pass\n"
        parsed_ast = ast.parse(mock_file_content)

        with patch("os.path.dirname", return_value="/fake/src"):
            with patch("os.path.join", return_value=plugin_dir):
                with patch("os.path.exists", return_value=True):
                    with patch("os.listdir", return_value=[plugin_filename]):
                        with patch(
                            "builtins.open", mock_open(read_data=mock_file_content)
                        ):
                            with patch("ast.parse", return_value=parsed_ast):
                                result = _check_llm_exists(llm_type)
                                assert result is True

    def test_check_action_exists_true(self):
        action_name = "test_action"
        interface_path = f"/fake/src/actions/{action_name}/interface.py"

        with patch("os.path.dirname", return_value="/fake/src"):
            with patch("os.path.join", return_value=interface_path):
                with patch("os.path.exists", return_value=True):
                    result = _check_action_exists(action_name)
                    assert result is True

    def test_check_action_exists_false(self):
        action_name = "missing_action"
        interface_path = f"/fake/src/actions/{action_name}/interface.py"

        with patch("os.path.dirname", return_value="/fake/src"):
            with patch("os.path.join", return_value=interface_path):
                with patch("os.path.exists", return_value=False):
                    result = _check_action_exists(action_name)
                    assert result is False

    def test_check_api_key_no_key_warning(self):
        raw_config = {}
        captured_output = []

        def mock_print(*args, **kwargs):
            captured_output.append(" ".join(map(str, args)))

        with patch("os.environ.get", return_value=""):
            with patch("builtins.print", side_effect=mock_print):
                _check_api_key(raw_config, verbose=False)
        assert any("Warning: No API key configured" in out for out in captured_output)

    def test_check_api_key_env_key_present(self):
        raw_config = {"api_key": "openmind_free"}
        captured_output = []

        def mock_print(*args, **kwargs):
            captured_output.append(" ".join(map(str, args)))

        with patch("os.environ.get", return_value="secret_key_from_env"):
            with patch("builtins.print", side_effect=mock_print):
                _check_api_key(raw_config, verbose=True)
        assert any(
            "API key configured (from environment)" in out for out in captured_output
        )

    def test_print_config_summary_single_mode(self):
        raw_config = {
            "name": "Test Config",
            "hertz": 10,
            "agent_inputs": ["inp1"],
            "agent_actions": ["act1"],
        }
        captured_output = []

        def mock_print(*args, **kwargs):
            captured_output.append(" ".join(map(str, args)))

        with patch("builtins.print", side_effect=mock_print):
            _print_config_summary(raw_config, is_multi_mode=False)
        assert any("Type: Single-mode" in out for out in captured_output)

    def test_print_config_summary_multi_mode(self):
        raw_config = {
            "name": "Test Multi Config",
            "default_mode": "default",
            "modes": {"mode1": {}},
            "transition_rules": [{"rule": 1}],
        }
        captured_output = []

        def mock_print(*args, **kwargs):
            captured_output.append(" ".join(map(str, args)))

        with patch("builtins.print", side_effect=mock_print):
            _print_config_summary(raw_config, is_multi_mode=True)
        assert any("Type: Multi-mode" in out for out in captured_output)

    def test_validate_components_success(self):
        raw_config = {
            "name": "Test",
            "agent_inputs": [{"type": "MockInput"}],
            "agent_actions": [{"name": "test_action"}],
            "cortex_llm": {"type": "MockLLM"},
        }
        with patch("src.cli._check_input_exists", return_value=True):
            with patch("src.cli._check_action_exists", return_value=True):
                with patch("src.cli._check_llm_exists", return_value=True):
                    _validate_components(raw_config, is_multi_mode=False, verbose=False)

    def test_validate_components_failure(self):
        raw_config = {
            "name": "Test",
            "agent_inputs": [{"type": "MissingInput"}],
        }
        with patch("src.cli._check_input_exists", return_value=False):
            with pytest.raises(ValueError, match="Component validation failed"):
                _validate_components(
                    raw_config, is_multi_mode=False, verbose=False, allow_missing=False
                )

    def test_validate_components_allow_missing(self):
        raw_config = {
            "name": "Test",
            "agent_inputs": [{"type": "MissingInput"}],
        }
        with patch("src.cli._check_input_exists", return_value=False):
            with patch("builtins.print") as mock_print:
                _validate_components(
                    raw_config, is_multi_mode=False, verbose=False, allow_missing=True
                )
                assert any(
                    "warnings:" in str(call) for call in mock_print.call_args_list
                )
