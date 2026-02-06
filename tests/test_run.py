import os
from unittest.mock import MagicMock, patch

import pytest
import typer

from run import app, setup_config_file, start


def test_setup_with_provided_config_name(tmp_path):
    """Test setup with a provided config name."""
    with patch("os.path.dirname", return_value=str(tmp_path)):
        config_dir = tmp_path / ".." / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_name = "test_config"
        config_name_result, config_path = setup_config_file(config_name)

        assert config_name_result == "test_config"
        assert config_path.endswith("test_config.json5")


def test_setup_with_no_config_name_uses_runtime(tmp_path):
    """Test setup without config name uses .runtime.json5."""
    with patch("os.path.dirname", return_value=str(tmp_path)):
        memory_dir = tmp_path / ".." / "config" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        runtime_file = memory_dir / ".runtime.json5"
        runtime_file.write_text('{"name": "runtime"}')

        config_dir = tmp_path / ".." / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        with patch("shutil.copy2") as mock_copy:
            config_name_result, config_path = setup_config_file(None)

            assert config_name_result == ".runtime"
            assert config_path.endswith(".runtime.json5")
            mock_copy.assert_called_once()


def test_setup_without_config_raises_when_runtime_missing(tmp_path):
    """Test that missing .runtime.json5 raises error."""
    with (
        patch("os.path.dirname", return_value=str(tmp_path)),
        patch("os.path.exists", return_value=False),
    ):
        with pytest.raises(typer.Exit):
            setup_config_file(None)


def test_setup_copies_runtime_config(tmp_path):
    """Test that runtime config is copied correctly."""
    with patch("os.path.dirname", return_value=str(tmp_path)):
        memory_dir = tmp_path / ".." / "config" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        runtime_file = memory_dir / ".runtime.json5"
        runtime_content = '{"name": "runtime", "hertz": 10}'
        runtime_file.write_text(runtime_content)

        config_dir = tmp_path / ".." / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        _, config_path = setup_config_file(None)

        assert os.path.exists(config_path)
        with open(config_path, "r") as f:
            content = f.read()
            assert content == runtime_content


def test_start_with_multi_mode_config():
    """Test starting with multi-mode configuration."""
    with (
        patch("run.setup_logging") as mock_setup_logging,
        patch("run.setup_config_file") as mock_setup_config,
        patch("run.json5.load") as mock_json5_load,
        patch("builtins.open"),
        patch("run.load_mode_config") as mock_load_mode_config,
        patch("run.ModeCortexRuntime") as mock_runtime_class,
        patch("asyncio.run") as mock_asyncio_run,
    ):

        # Setup mocks
        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_json5_load.return_value = {"modes": {}, "default_mode": "test"}

        mock_mode_config = MagicMock()
        mock_mode_config.modes = {"mode1": MagicMock()}
        mock_mode_config.default_mode = "mode1"
        mock_load_mode_config.return_value = mock_mode_config

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        start(
            config_name="test_config",
            hot_reload=True,
            check_interval=60,
            log_level="INFO",
            log_to_file=False,
        )

        mock_setup_logging.assert_called_once_with("test_config", "INFO", False)
        mock_load_mode_config.assert_called_once_with("test_config")
        mock_runtime_class.assert_called_once_with(
            mock_mode_config, "test_config", hot_reload=True, check_interval=60
        )
        mock_asyncio_run.assert_called_once()


def test_start_with_single_mode_config():
    """Test starting with single-mode configuration."""
    with (
        patch("run.setup_logging") as mock_setup_logging,
        patch("run.setup_config_file") as mock_setup_config,
        patch("run.json5.load") as mock_json5_load,
        patch("builtins.open"),
        patch("run.load_config") as mock_load_config,
        patch("run.CortexRuntime") as mock_runtime_class,
        patch("asyncio.run") as mock_asyncio_run,
    ):
        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_json5_load.return_value = {"name": "test"}

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        start(
            config_name="test_config",
            hot_reload=True,
            check_interval=60,
            log_level="INFO",
            log_to_file=False,
        )

        mock_setup_logging.assert_called_once_with("test_config", "INFO", False)
        mock_load_config.assert_called_once_with("test_config")
        mock_runtime_class.assert_called_once_with(
            mock_config, "test_config", hot_reload=True, check_interval=60
        )
        mock_asyncio_run.assert_called_once()


def test_start_with_file_not_found():
    """Test start command with missing config file."""
    with (
        patch("run.setup_logging"),
        patch("run.setup_config_file") as mock_setup_config,
        patch("builtins.open") as mock_open,
    ):

        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_open.side_effect = FileNotFoundError()

        with pytest.raises(typer.Exit):
            start(
                config_name="test_config",
                hot_reload=True,
                check_interval=60,
                log_level="INFO",
                log_to_file=False,
            )


def test_start_with_generic_exception():
    """Test start command with generic exception."""
    with (
        patch("run.setup_logging"),
        patch("run.setup_config_file") as mock_setup_config,
        patch("run.json5.load") as mock_json5_load,
        patch("builtins.open"),
        patch("run.load_config") as mock_load_config,
    ):

        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_json5_load.return_value = {"name": "test"}
        mock_load_config.side_effect = Exception("Test error")

        with pytest.raises(typer.Exit):
            start(
                config_name="test_config",
                hot_reload=True,
                check_interval=60,
                log_level="INFO",
                log_to_file=False,
            )


def test_start_with_hot_reload_disabled():
    """Test starting with hot reload disabled."""
    with (
        patch("run.setup_logging"),
        patch("run.setup_config_file") as mock_setup_config,
        patch("run.json5.load") as mock_json5_load,
        patch("builtins.open"),
        patch("run.load_config") as mock_load_config,
        patch("run.CortexRuntime") as mock_runtime_class,
        patch("asyncio.run"),
    ):

        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_json5_load.return_value = {"name": "test"}

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        start(
            config_name="test_config",
            hot_reload=False,
            check_interval=60,
            log_level="INFO",
            log_to_file=False,
        )

        mock_runtime_class.assert_called_once_with(
            mock_config, "test_config", hot_reload=False, check_interval=60
        )


def test_start_with_custom_check_interval():
    """Test starting with custom check interval."""
    with (
        patch("run.setup_logging"),
        patch("run.setup_config_file") as mock_setup_config,
        patch("run.json5.load") as mock_json5_load,
        patch("builtins.open"),
        patch("run.load_config") as mock_load_config,
        patch("run.CortexRuntime") as mock_runtime_class,
        patch("asyncio.run"),
    ):

        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_json5_load.return_value = {"name": "test"}

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        start(
            config_name="test_config",
            hot_reload=True,
            check_interval=120,
            log_level="INFO",
            log_to_file=False,
        )

        mock_runtime_class.assert_called_once_with(
            mock_config, "test_config", hot_reload=True, check_interval=120
        )


def test_start_with_custom_log_level():
    """Test starting with custom log level."""
    with (
        patch("run.setup_logging") as mock_setup_logging,
        patch("run.setup_config_file") as mock_setup_config,
        patch("run.json5.load") as mock_json5_load,
        patch("builtins.open"),
        patch("run.load_config") as mock_load_config,
        patch("run.CortexRuntime") as mock_runtime_class,
        patch("asyncio.run"),
    ):

        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_json5_load.return_value = {"name": "test"}

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        start(
            config_name="test_config",
            hot_reload=True,
            check_interval=60,
            log_level="DEBUG",
            log_to_file=False,
        )

        mock_setup_logging.assert_called_once_with("test_config", "DEBUG", False)


def test_start_with_log_to_file():
    """Test starting with log to file enabled."""
    with (
        patch("run.setup_logging") as mock_setup_logging,
        patch("run.setup_config_file") as mock_setup_config,
        patch("run.json5.load") as mock_json5_load,
        patch("builtins.open"),
        patch("run.load_config") as mock_load_config,
        patch("run.CortexRuntime") as mock_runtime_class,
        patch("asyncio.run"),
    ):

        mock_setup_config.return_value = ("test_config", "/path/to/test_config.json5")
        mock_json5_load.return_value = {"name": "test"}

        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        mock_runtime = MagicMock()
        mock_runtime_class.return_value = mock_runtime

        start(
            config_name="test_config",
            hot_reload=True,
            check_interval=60,
            log_level="INFO",
            log_to_file=True,
        )

        mock_setup_logging.assert_called_once_with("test_config", "INFO", True)


def test_start_without_config_name_uses_default():
    """Test that start without config_name calls setup_config_file with None."""
    with (
        patch("run.setup_logging"),
        patch("run.setup_config_file") as mock_setup_config,
    ):

        mock_setup_config.return_value = (".runtime", "/path/to/.runtime.json5")
        mock_setup_config.side_effect = typer.Exit(1)

        with pytest.raises(typer.Exit):
            start(
                config_name=None,
                hot_reload=True,
                check_interval=60,
                log_level="INFO",
                log_to_file=False,
            )

        mock_setup_config.assert_called_once_with(None)


def test_app_is_typer_instance():
    """Test that app is a Typer instance."""
    assert isinstance(app, typer.Typer)


def test_start_command_exists():
    """Test that start command is registered."""
    assert hasattr(app, "registered_commands") or hasattr(app, "registered_groups")
