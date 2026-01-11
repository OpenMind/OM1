"""Doctor command - System health check for OM1."""

from typing import Optional

import typer
from rich.panel import Panel

from cli.utils.output import console
from cli.utils.system import CheckStatus, run_all_checks


def doctor(
    config_name: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Check requirements for specific config.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed information for each check.",
    ),
) -> None:
    """
    Check system requirements and diagnose issues.

    Verifies Python version, dependencies, environment variables,
    and external services needed to run OM1 agents.

    Examples
    --------
        om1 doctor                    # General health check
        om1 doctor --config ollama    # Check for specific config
        om1 doctor --verbose          # Detailed output
    """
    title = "OM1 System Health Check"
    if config_name:
        title += f" (for {config_name})"

    console.print(Panel(f"[bold]{title}[/bold]", border_style="blue"))
    console.print()

    results = run_all_checks(config_name)

    # Count by status
    ok_count = sum(1 for r in results if r.status == CheckStatus.OK)
    warn_count = sum(1 for r in results if r.status == CheckStatus.WARN)
    fail_count = sum(1 for r in results if r.status == CheckStatus.FAIL)

    # Display results
    for result in results:
        if result.status == CheckStatus.OK:
            status_icon = "[green][OK][/green]"
        elif result.status == CheckStatus.WARN:
            status_icon = "[yellow][WARN][/yellow]"
        else:
            status_icon = "[red][FAIL][/red]"

        console.print(f"  {status_icon} {result.name}: {result.message}")

        if verbose and result.details:
            console.print(f"       [dim]{result.details}[/dim]")

        if result.status != CheckStatus.OK and result.fix_hint:
            console.print(f"       [dim]Fix: {result.fix_hint}[/dim]")

    # Summary
    console.print()
    if fail_count == 0 and warn_count == 0:
        console.print("[bold green]All checks passed![/bold green]")
        if config_name:
            console.print(f"\nReady to run: [cyan]om1 run {config_name}[/cyan]")
    else:
        summary_parts = []
        if ok_count > 0:
            summary_parts.append(f"[green]{ok_count} passed[/green]")
        if warn_count > 0:
            summary_parts.append(f"[yellow]{warn_count} warnings[/yellow]")
        if fail_count > 0:
            summary_parts.append(f"[red]{fail_count} failed[/red]")

        console.print(f"Results: {', '.join(summary_parts)}")

        if fail_count > 0:
            console.print(
                "\n[red]Please fix the failed checks before running OM1.[/red]"
            )
            raise typer.Exit(1)
        else:
            console.print(
                "\n[yellow]Warnings may cause issues with some features.[/yellow]"
            )
