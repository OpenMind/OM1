"""Stop command - Stop running OM1 agents."""

from typing import Optional

import typer

from cli.utils.output import console, print_error, print_info, print_success
from cli.utils.process import get_running_agents, stop_agent, stop_all_agents


def stop(
    config_name: Optional[str] = typer.Argument(
        None,
        help="Config name to stop. Use --all to stop all agents.",
    ),
    all_agents: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Stop all running agents.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force kill (SIGKILL) instead of graceful shutdown.",
    ),
    timeout: int = typer.Option(
        5,
        "--timeout",
        "-t",
        help="Seconds to wait for graceful shutdown before giving up.",
    ),
) -> None:
    """
    Stop running OM1 agents gracefully.

    Sends SIGTERM for graceful shutdown. Use --force for immediate termination.

    Examples
    --------
        om1 stop ollama           # Stop specific agent
        om1 stop --all            # Stop all agents
        om1 stop ollama --force   # Force kill
        om1 stop --all --timeout 10
    """
    if all_agents:
        agents = get_running_agents()
        if not agents:
            print_info("No agents currently running")
            return

        console.print(f"Stopping [bold]{len(agents)}[/bold] agent(s)...\n")
        results = stop_all_agents(force=force, timeout=timeout)

        success_count = 0
        for cfg_name, success, message in results:
            if success:
                print_success(message)
                success_count += 1
            else:
                print_error(message)

        console.print(f"\nStopped [bold]{success_count}/{len(results)}[/bold] agent(s)")
        return

    if config_name is None:
        print_error("Please specify a config name or use --all")
        console.print("\nUsage:")
        console.print("  om1 stop <config>  # Stop specific agent")
        console.print("  om1 stop --all     # Stop all agents")

        # Show running agents
        agents = get_running_agents()
        if agents:
            console.print("\nCurrently running:")
            for agent in agents:
                console.print(
                    f"  - [cyan]{agent['config']}[/cyan] (PID {agent['pid']})"
                )

        raise typer.Exit(1)

    # Stop specific agent
    success, message = stop_agent(config_name, force=force, timeout=timeout)
    if success:
        print_success(message)
    else:
        print_error(message)
        raise typer.Exit(1)
