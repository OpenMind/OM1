"""List command - List available configurations."""

import typer
from rich.table import Table

from cli.utils.config import list_configs as get_configs
from cli.utils.output import console


def list_configs(
    mode_aware: bool = typer.Option(
        False,
        "--mode-aware",
        "-m",
        help="Show only mode-aware configurations.",
    ),
    standard: bool = typer.Option(
        False,
        "--standard",
        "-s",
        help="Show only standard configurations.",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, simple, json.",
    ),
) -> None:
    """
    List all available configuration files.

    Displays configurations categorized as mode-aware (multi-mode)
    or standard (single-mode).

    Examples
    --------
        om1 list
        om1 list --mode-aware
        om1 list --format simple
    """
    standard_configs, mode_configs = get_configs()

    if not standard_configs and not mode_configs:
        console.print("[yellow]No configuration files found.[/yellow]")
        return

    # Filter based on options
    show_standard = not mode_aware or (not mode_aware and not standard)
    show_mode = not standard or (not mode_aware and not standard)

    if mode_aware and not standard:
        show_standard = False
        show_mode = True
    elif standard and not mode_aware:
        show_standard = True
        show_mode = False

    if format == "json":
        import json

        output = {}
        if show_mode and mode_configs:
            output["mode_aware"] = [
                {"name": name, "display_name": display}
                for name, display in mode_configs
            ]
        if show_standard and standard_configs:
            output["standard"] = [
                {"name": name, "display_name": display}
                for name, display in standard_configs
            ]
        console.print(json.dumps(output, indent=2))
        return

    if format == "simple":
        if show_mode and mode_configs:
            console.print("[bold]Mode-Aware:[/bold]")
            for name, display in mode_configs:
                console.print(f"  {name}")
        if show_standard and standard_configs:
            console.print("[bold]Standard:[/bold]")
            for name, display in standard_configs:
                console.print(f"  {name}")
        return

    # Default: table format
    if show_mode and mode_configs:
        table = Table(title="Mode-Aware Configurations", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="green")
        for name, display in mode_configs:
            table.add_row(name, display)
        console.print(table)
        console.print()

    if show_standard and standard_configs:
        table = Table(title="Standard Configurations", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="green")
        for name, display in standard_configs:
            table.add_row(name, display)
        console.print(table)
