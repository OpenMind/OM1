"""Run command - Start an OM1 agent with configuration."""

import asyncio
import logging
import shutil
from typing import Optional

import typer

from cli.utils.config import get_config_dir
from cli.utils.output import console, print_error, print_info, print_warning
from cli.utils.process import (
    is_agent_running,
    register_cleanup_handler,
    remove_pid_file,
    remove_state,
    write_pid_file,
    write_state,
)


def setup_config_file(config_name: Optional[str]) -> tuple:
    """
    Set up the configuration file.

    Parameters
    ----------
    config_name : str, optional
        The name of the configuration file (without extension).
        If not provided, uses .runtime.json5 from memory folder.

    Returns
    -------
    tuple
        (config_name, config_path)
    """
    config_dir = get_config_dir()

    if config_name is None:
        runtime_config_path = config_dir / "memory" / ".runtime.json5"

        if not runtime_config_path.exists():
            print_error(
                f"Default runtime configuration not found: {runtime_config_path}"
            )
            print_info(
                "Please provide a config_name or ensure .runtime.json5 exists in config/memory/"
            )
            raise typer.Exit(1)

        config_name = ".runtime"
        config_path = config_dir / f"{config_name}.json5"

        shutil.copy2(runtime_config_path, config_path)
        print_info("Using default runtime configuration from memory folder")

    else:
        config_path = config_dir / f"{config_name}.json5"

        if not config_path.exists():
            print_error(f"Configuration file not found: {config_path}")
            raise typer.Exit(1)

    return config_name, str(config_path)


def run(
    config_name: Optional[str] = typer.Argument(
        None,
        help="Configuration file name (without .json5 extension). "
        "If not provided, uses .runtime.json5 from memory folder.",
    ),
    hot_reload: bool = typer.Option(
        True,
        "--hot-reload/--no-hot-reload",
        help="Enable hot-reload of configuration files.",
    ),
    check_interval: int = typer.Option(
        60,
        "--check-interval",
        help="Interval in seconds between config file checks (for hot-reload).",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    ),
    log_to_file: bool = typer.Option(
        False,
        "--log-to-file",
        help="Save logs to file in logs/ directory.",
    ),
) -> None:
    """
    Start an OM1 agent with specified configuration.

    Examples
    --------
        om1 run spot
        om1 run ollama --log-level DEBUG
        om1 run test --no-hot-reload
    """
    # First validate config exists before loading heavy dependencies
    config_name, config_path = setup_config_file(config_name)

    # Check if agent is already running
    if is_agent_running(config_name):
        print_warning(f"Agent '{config_name}' is already running")
        print_info(
            "Use 'om1 stop {config_name}' to stop it first, or 'om1 status' to check"
        )
        raise typer.Exit(1)

    # Lazy imports to avoid loading heavy dependencies at CLI startup
    import json5

    from runtime.logging import setup_logging
    from runtime.multi_mode.config import load_mode_config
    from runtime.multi_mode.cortex import ModeCortexRuntime
    from runtime.single_mode.config import load_config
    from runtime.single_mode.cortex import CortexRuntime

    setup_logging(config_name, log_level, log_to_file)

    # Write PID and state files
    write_pid_file(config_name)
    register_cleanup_handler(config_name)

    try:
        with open(config_path, "r") as f:
            raw_config = json5.load(f)

        is_multi_mode = "modes" in raw_config and "default_mode" in raw_config
        current_mode = "default"

        if is_multi_mode:
            mode_config = load_mode_config(config_name)
            runtime = ModeCortexRuntime(
                mode_config,
                config_name,
                hot_reload=hot_reload,
                check_interval=check_interval,
            )
            current_mode = mode_config.default_mode
            console.print(
                f"[green]Starting OM1[/green] with mode-aware configuration: "
                f"[cyan]{config_name}[/cyan]"
            )
            console.print(f"  Available modes: {list(mode_config.modes.keys())}")
            console.print(
                f"  Default mode: [yellow]{mode_config.default_mode}[/yellow]"
            )
        else:
            config = load_config(config_name)
            runtime = CortexRuntime(
                config,
                config_name,
                hot_reload=hot_reload,
                check_interval=check_interval,
            )
            console.print(
                f"[green]Starting OM1[/green] with configuration: [cyan]{config_name}[/cyan]"
            )

        # Write state file with runtime info
        write_state(
            config_name,
            mode=current_mode,
            hot_reload_enabled=hot_reload,
            extra_data={"is_multi_mode": is_multi_mode},
        )

        if hot_reload:
            print_info(f"Hot-reload enabled (check interval: {check_interval}s)")

        asyncio.run(runtime.run())

    except FileNotFoundError:
        print_error(f"Configuration file not found: {config_path}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        logging.exception("Detailed error:")
        raise typer.Exit(1)
    finally:
        # Clean up PID and state files
        remove_pid_file(config_name)
        remove_state(config_name)
