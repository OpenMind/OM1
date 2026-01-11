"""Modes command - Show detailed mode information for mode-aware configurations."""

import json
import logging
from typing import Any, Dict

import typer
from rich.panel import Panel
from rich.table import Table

from cli.utils.config import is_mode_aware_config, resolve_config_path
from cli.utils.output import console, print_error, print_warning


def modes(
    config_name: str = typer.Argument(
        ...,
        help="Configuration file name (without .json5 extension).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
) -> None:
    """
    Show detailed information about modes in a mode-aware configuration.

    Displays available modes, transition rules, and settings for multi-mode
    configurations. Useful for debugging and understanding mode system behavior.

    Examples
    --------
        om1 modes spot_modes
        om1 modes unitree_go2_modes
        om1 modes spot_modes --json
    """
    # First check if config exists
    try:
        config_path = resolve_config_path(config_name)
    except FileNotFoundError:
        print_error(f"Configuration file not found: {config_name}.json5")
        raise typer.Exit(1)

    # Check if it's a mode-aware config
    if not is_mode_aware_config(config_path):
        print_error(
            f"Configuration '{config_name}' is not mode-aware.\n"
            "   Use 'om1 info' for standard configurations."
        )
        print_warning("Mode-aware configs must have 'modes' and 'default_mode' fields.")
        raise typer.Exit(1)

    try:
        from runtime.multi_mode.config import load_mode_config

        mode_config = load_mode_config(config_name)
    except ModuleNotFoundError as e:
        missing_module = str(e).replace("No module named ", "").strip("'")
        print_error(f"Missing dependency: {missing_module}")
        print_warning(
            "This configuration requires additional dependencies.\n"
            "   Try: uv sync --extra robotics"
        )
        raise typer.Exit(1)
    except FileNotFoundError:
        print_error(f"Configuration file not found: {config_name}.json5")
        raise typer.Exit(1)
    except Exception as e:
        logging.exception("Error loading mode configuration")
        print_error(f"Error loading mode configuration: {e}")
        raise typer.Exit(1)

    # JSON output
    if json_output:
        output: Dict[str, Any] = {
            "name": mode_config.name,
            "default_mode": mode_config.default_mode,
            "allow_manual_switching": mode_config.allow_manual_switching,
            "mode_memory_enabled": mode_config.mode_memory_enabled,
            "modes": {},
            "transition_rules": [],
        }

        for name, mode in mode_config.modes.items():
            output["modes"][name] = {
                "display_name": mode.display_name,
                "description": mode.description,
                "hertz": mode.hertz,
                "timeout_seconds": mode.timeout_seconds,
                "inputs_count": len(mode._raw_inputs),
                "actions_count": len(mode._raw_actions),
                "lifecycle_hooks_count": (
                    len(mode.lifecycle_hooks) if mode.lifecycle_hooks else 0
                ),
            }

        for rule in mode_config.transition_rules:
            output["transition_rules"].append(
                {
                    "from_mode": rule.from_mode,
                    "to_mode": rule.to_mode,
                    "type": rule.transition_type.value,
                    "trigger_keywords": rule.trigger_keywords,
                    "priority": rule.priority,
                    "cooldown_seconds": rule.cooldown_seconds,
                }
            )

        console.print(json.dumps(output, indent=2))
        return

    # Rich table output
    # Header
    console.print(
        Panel(
            f"[bold]Mode System: {mode_config.name}[/bold]",
            border_style="blue",
        )
    )
    console.print()

    # General info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Property", style="dim")
    info_table.add_column("Value")

    info_table.add_row("Default Mode", f"[yellow]{mode_config.default_mode}[/yellow]")
    info_table.add_row(
        "Manual Switching",
        (
            "[green]Enabled[/green]"
            if mode_config.allow_manual_switching
            else "[red]Disabled[/red]"
        ),
    )
    info_table.add_row(
        "Mode Memory",
        (
            "[green]Enabled[/green]"
            if mode_config.mode_memory_enabled
            else "[red]Disabled[/red]"
        ),
    )
    if mode_config.global_lifecycle_hooks:
        info_table.add_row(
            "Global Lifecycle Hooks",
            str(len(mode_config.global_lifecycle_hooks)),
        )

    console.print(info_table)
    console.print()

    # Available modes
    console.print("[bold]Available Modes:[/bold]")
    console.print()

    modes_table = Table(show_header=True)
    modes_table.add_column("Name", style="cyan")
    modes_table.add_column("Display Name")
    modes_table.add_column("Frequency")
    modes_table.add_column("Timeout")
    modes_table.add_column("Inputs", justify="right")
    modes_table.add_column("Actions", justify="right")

    for name, mode in mode_config.modes.items():
        is_default = (
            " [yellow](DEFAULT)[/yellow]" if name == mode_config.default_mode else ""
        )
        timeout_str = f"{mode.timeout_seconds}s" if mode.timeout_seconds else "-"

        modes_table.add_row(
            name + is_default,
            mode.display_name,
            f"{mode.hertz} Hz",
            timeout_str,
            str(len(mode._raw_inputs)),
            str(len(mode._raw_actions)),
        )

    console.print(modes_table)
    console.print()

    # Mode details
    for name, mode in mode_config.modes.items():
        is_default = " (DEFAULT)" if name == mode_config.default_mode else ""
        console.print(f"[bold cyan]{mode.display_name}[/bold cyan]{is_default}")
        console.print(f"  [dim]Name:[/dim] {name}")
        console.print(f"  [dim]Description:[/dim] {mode.description}")

        if mode.lifecycle_hooks:
            console.print(f"  [dim]Lifecycle Hooks:[/dim] {len(mode.lifecycle_hooks)}")

        console.print()

    # Transition rules
    if mode_config.transition_rules:
        console.print("[bold]Transition Rules:[/bold]")
        console.print()

        rules_table = Table(show_header=True)
        rules_table.add_column("From", style="cyan")
        rules_table.add_column("To", style="green")
        rules_table.add_column("Type")
        rules_table.add_column("Keywords")
        rules_table.add_column("Priority", justify="right")
        rules_table.add_column("Cooldown")

        for rule in mode_config.transition_rules:
            from_display = (
                mode_config.modes[rule.from_mode].display_name
                if rule.from_mode != "*"
                else "[dim]Any Mode[/dim]"
            )
            to_display = mode_config.modes[rule.to_mode].display_name

            keywords = (
                ", ".join(rule.trigger_keywords) if rule.trigger_keywords else "-"
            )
            cooldown = f"{rule.cooldown_seconds}s" if rule.cooldown_seconds > 0 else "-"

            rules_table.add_row(
                from_display,
                to_display,
                rule.transition_type.value,
                keywords,
                str(rule.priority),
                cooldown,
            )

        console.print(rules_table)
