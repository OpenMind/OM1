"""Status command - Show running OM1 agents."""

import typer
from rich.table import Table

from cli.utils.output import console, print_info
from cli.utils.process import get_running_agents


def status(
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Continuously refresh status (not implemented yet).",
    ),
) -> None:
    """
    Show status of running OM1 agents.

    Displays all running agents with their PID, config name, mode, and uptime.

    Examples
    --------
        om1 status
        om1 status --watch
    """
    agents = get_running_agents()

    if not agents:
        print_info("No OM1 agents currently running")
        console.print("\nTo start an agent: [cyan]om1 run <config>[/cyan]")
        console.print("To list configs:   [cyan]om1 list[/cyan]")
        return

    table = Table(title="OM1 Running Agents", show_header=True)
    table.add_column("PID", style="cyan", justify="right")
    table.add_column("Config", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Uptime", justify="right")
    table.add_column("Hot Reload")
    table.add_column("Status", style="bold")

    for agent in agents:
        hot_reload_status = (
            "[green]On[/green]" if agent["hot_reload"] else "[dim]Off[/dim]"
        )
        table.add_row(
            str(agent["pid"]),
            agent["config"],
            agent["mode"],
            agent["uptime"],
            hot_reload_status,
            f"[green]{agent['status']}[/green]",
        )

    console.print(table)
    console.print(f"\nTotal: [bold]{len(agents)}[/bold] agent(s) running")
    console.print(
        "\nTo stop: [cyan]om1 stop <config>[/cyan] or [cyan]om1 stop --all[/cyan]"
    )
