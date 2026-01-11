"""Rich console output utilities for CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_success(message: str) -> None:
    """Print a success message with green checkmark."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message with red X."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow exclamation."""
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message with blue dot."""
    console.print(f"[blue]•[/blue] {message}")


def create_table(title: str, columns: list) -> Table:
    """
    Create a Rich table with predefined styling.

    Parameters
    ----------
    title : str
        Table title
    columns : list
        List of column names

    Returns
    -------
    Table
        Configured Rich Table
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col in columns:
        table.add_column(col)
    return table


def create_panel(content: str, title: str, style: str = "blue") -> Panel:
    """
    Create a Rich panel with predefined styling.

    Parameters
    ----------
    content : str
        Panel content
    title : str
        Panel title
    style : str
        Border style color

    Returns
    -------
    Panel
        Configured Rich Panel
    """
    return Panel(content, title=title, border_style=style)
