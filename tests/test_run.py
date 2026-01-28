"""
Unit tests for the main application entry point (src/run.py).
Tests the 'start' command logic directly by invoking the function.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# --- Setup path *before* importing from src ---
current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
# ------------------------------------------------

from src.run import (  # noqa: E402 (Ignore E402 for this specific case where path setup is required before import)
    start,
)


class TestStartCommand:

    def test_start_command_single_mode(self):
        config_name = "test_single"
        fake_config_path = f"/fake/path/{config_name}.json5"

        with patch(
            "src.run.setup_config_file", return_value=(config_name, fake_config_path)
        ) as mock_setup_conf:
            with patch("src.run.setup_logging") as mock_setup_log:
                raw_config_content = {
                    "name": config_name,
                    "version": "v1.0.1",
                    "hertz": 10.0,
                    "agent_inputs": [],
                    "agent_actions": [],
                }
                with patch(
                    "builtins.open",
                    mock_open(
                        read_data='{"name": "test", "version": "v1.0.1", "hertz": 10.0, "agent_inputs": [], "agent_actions": []}'
                    ),
                ):
                    with patch("src.run.json5.load", return_value=raw_config_content):
                        with patch("src.run.load_config") as mock_load_config:
                            mock_config_obj = MagicMock()
                            mock_load_config.return_value = mock_config_obj

                            with patch("src.run.CortexRuntime") as MockRuntimeClass:
                                mock_runtime_instance = MagicMock()
                                MockRuntimeClass.return_value = mock_runtime_instance

                                with patch("src.run.asyncio.run") as mock_async_run:
                                    start(
                                        config_name=config_name,
                                        hot_reload=True,
                                        check_interval=60,
                                        log_level="INFO",
                                        log_to_file=False,
                                    )

                                    mock_setup_conf.assert_called_once_with(config_name)
                                    mock_setup_log.assert_called_once_with(
                                        config_name, "INFO", False
                                    )
                                    mock_load_config.assert_called_once_with(
                                        config_name
                                    )
                                    MockRuntimeClass.assert_called_once_with(
                                        mock_config_obj,
                                        config_name,
                                        hot_reload=True,
                                        check_interval=60,
                                    )
                                    mock_runtime_instance.run.assert_called_once()
                                    mock_async_run.assert_called_once()

    def test_start_command_multi_mode(self):
        config_name = "test_multi"
        fake_config_path = f"/fake/path/{config_name}.json5"

        with patch(
            "src.run.setup_config_file", return_value=(config_name, fake_config_path)
        ) as mock_setup_conf:
            with patch("src.run.setup_logging") as mock_setup_log:
                raw_config_content = {
                    "name": config_name,
                    "version": "v1.0.1",
                    "modes": {"mode1": {}},
                    "default_mode": "mode1",
                }
                with patch(
                    "builtins.open",
                    mock_open(
                        read_data='{"name": "test", "version": "v1.0.1", "modes": {"mode1": {}}, "default_mode": "mode1"}'
                    ),
                ):
                    with patch("src.run.json5.load", return_value=raw_config_content):
                        with patch("src.run.load_mode_config") as mock_load_mode_config:
                            mock_mode_config_obj = MagicMock()
                            mock_load_mode_config.return_value = mock_mode_config_obj

                            with patch("src.run.ModeCortexRuntime") as MockRuntimeClass:
                                mock_runtime_instance = MagicMock()
                                MockRuntimeClass.return_value = mock_runtime_instance

                                with patch("src.run.asyncio.run") as mock_async_run:
                                    start(
                                        config_name=config_name,
                                        hot_reload=True,
                                        check_interval=60,
                                        log_level="INFO",
                                        log_to_file=False,
                                    )

                                    mock_setup_conf.assert_called_once_with(config_name)
                                    mock_setup_log.assert_called_once_with(
                                        config_name, "INFO", False
                                    )
                                    mock_load_mode_config.assert_called_once_with(
                                        config_name
                                    )
                                    MockRuntimeClass.assert_called_once_with(
                                        mock_mode_config_obj,
                                        config_name,
                                        hot_reload=True,
                                        check_interval=60,
                                    )
                                    mock_runtime_instance.run.assert_called_once()
                                    mock_async_run.assert_called_once()

    def test_start_command_config_not_found(self):
        config_name = "non_existent"
        fake_config_path = f"/fake/path/{config_name}.json5"

        with patch(
            "src.run.setup_config_file", return_value=(config_name, fake_config_path)
        ):
            with patch("src.run.setup_logging"):
                with patch(
                    "builtins.open",
                    side_effect=FileNotFoundError(
                        f"Config file not found: {fake_config_path}"
                    ),
                ):
                    with patch("src.run.json5.load") as mock_json5_load:
                        with (
                            patch("src.run.load_config") as mock_load_config,
                            patch("src.run.load_mode_config") as mock_load_mode_config,
                        ):
                            with patch("src.run.asyncio.run") as mock_async_run:
                                exception_raised = False
                                try:
                                    start(
                                        config_name=config_name,
                                        hot_reload=True,
                                        check_interval=60,
                                        log_level="INFO",
                                        log_to_file=False,
                                    )
                                except SystemExit as e:
                                    if e.code == 1:
                                        exception_raised = True
                                except Exception as e:
                                    try:
                                        import click

                                        if (
                                            isinstance(e, click.exceptions.Exit)
                                            and e.exit_code == 1
                                        ):
                                            exception_raised = True
                                        else:
                                            raise
                                    except ImportError:
                                        pass

                                assert (
                                    exception_raised
                                ), "Expected an Exit exception with code 1"

                                mock_json5_load.assert_not_called()
                                mock_load_config.assert_not_called()
                                mock_load_mode_config.assert_not_called()
                                mock_async_run.assert_not_called()
