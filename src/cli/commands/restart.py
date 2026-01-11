"""Restart command - Restart running OM1 agents."""

import os
import subprocess
import sys
import time

import typer

from cli.utils.output import console, print_error, print_info, print_success
from cli.utils.process import is_agent_running, read_state, stop_agent


def restart(
    config_name: str = typer.Argument(
        ...,
        help="Config name to restart.",
    ),
    hot_reload: bool = typer.Option(
        False,
        "--hot-reload",
        "-r",
        help="Trigger hot-reload only (touch config file).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force kill before restart.",
    ),
    timeout: int = typer.Option(
        5,
        "--timeout",
        "-t",
        help="Seconds to wait for shutdown.",
    ),
) -> None:
    """
    Restart a running OM1 agent.

    With --hot-reload, triggers config hot-reload without full restart.
    Otherwise, stops the agent and starts it again.

    Examples
    --------
        om1 restart ollama              # Full restart
        om1 restart ollama --hot-reload # Trigger hot-reload only
        om1 restart ollama --force      # Force kill then start
    """
    if hot_reload:
        _trigger_hot_reload(config_name)
        return

    # Full restart
    if not is_agent_running(config_name):
        print_info(f"Agent '{config_name}' is not running")
        console.print(f"Starting fresh: [cyan]om1 run {config_name}[/cyan]")

        # Start the agent
        _start_agent(config_name)
        return

    # Get current state for restart
    state = read_state(config_name) or {}
    hot_reload_was_enabled = state.get("hot_reload_enabled", True)

    # Stop the agent
    console.print(f"Stopping [cyan]{config_name}[/cyan]...")
    success, message = stop_agent(config_name, force=force, timeout=timeout)

    if not success:
        print_error(message)
        raise typer.Exit(1)

    print_success(message)

    # Brief pause to ensure cleanup
    time.sleep(0.5)

    # Start the agent again
    console.print(f"\nStarting [cyan]{config_name}[/cyan]...")
    _start_agent(config_name, hot_reload=hot_reload_was_enabled)


def _trigger_hot_reload(config_name: str) -> None:
    """Trigger hot-reload by touching the config file."""
    from cli.utils.config import get_config_dir

    if not is_agent_running(config_name):
        print_error(f"Agent '{config_name}' is not running")
        print_info("Use 'om1 run' to start the agent first")
        raise typer.Exit(1)

    # Touch the runtime config file to trigger hot-reload
    config_dir = get_config_dir()
    runtime_config = config_dir / "memory" / ".runtime.json5"

    if runtime_config.exists():
        # Touch the file to update mtime
        os.utime(runtime_config, None)
        print_success(f"Hot-reload triggered for '{config_name}'")
        print_info("Config will be reloaded on next check interval")
    else:
        # Touch the main config file
        config_path = config_dir / f"{config_name}.json5"
        if config_path.exists():
            os.utime(config_path, None)
            print_success(f"Config file touched: {config_path}")
            print_info("Hot-reload will occur on next check interval")
        else:
            print_error(f"Config file not found: {config_path}")
            raise typer.Exit(1)


def _start_agent(config_name: str, hot_reload: bool = True) -> None:
    """Start an agent in the current process."""
    # Build the command
    args = [sys.executable, "-m", "cli.main", "run", config_name]
    if not hot_reload:
        args.append("--no-hot-reload")

    try:
        # Replace current process with the new agent
        # This ensures proper signal handling
        console.print(f"Running: [dim]om1 run {config_name}[/dim]\n")

        # Use subprocess to start in foreground
        subprocess.run(args, check=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
