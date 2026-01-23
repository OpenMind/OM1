"""Status command - Show running OM1 agents."""

import time

import typer
from rich.live import Live
from rich.table import Table

from cli.utils.output import console, print_info
from cli.utils.process import get_running_agents


def _build_status_table() -> tuple[Table, int]:
    """Build the status table and return it with agent count."""
    agents = get_running_agents()

    if not agents:
        table = Table(title="OM1 Running Agents", show_header=True)
        table.add_column("Status")
        table.add_row("[dim]No agents running[/dim]")
        return table, 0

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

    return table, len(agents)


def status(
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Continuously refresh status every 2 seconds.",
    ),
    interval: float = typer.Option(
        2.0,
        "--interval",
        "-i",
        help="Refresh interval in seconds (only with --watch).",
    ),
) -> None:
    """
    Show status of running OM1 agents.

    Displays all running agents with their PID, config name, mode, and uptime.

    Examples
    --------
        om1 status
        om1 status --watch
        om1 status --watch --interval 5
    """
    if watch:
        console.print("[dim]Press Ctrl+C to stop watching[/dim]\n")
        try:
            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    table, count = _build_status_table()
                    live.update(table)
                    time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped watching.[/yellow]")
            return

    # Non-watch mode: single display
    agents = get_running_agents()

    if not agents:
        print_info("No OM1 agents currently running")
        console.print("\nTo start an agent: [cyan]om1 run <config>[/cyan]")
        console.print("To list configs:   [cyan]om1 list[/cyan]")
        return

    table, count = _build_status_table()
    console.print(table)
    console.print(f"\nTotal: [bold]{count}[/bold] agent(s) running")
    console.print(
        "\nTo stop: [cyan]om1 stop <config>[/cyan] or [cyan]om1 stop --all[/cyan]"
    )
