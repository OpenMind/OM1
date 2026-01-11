"""Info command - Show configuration details."""

import typer
from rich.panel import Panel
from rich.table import Table

from cli.utils.config import (
    is_mode_aware_config,
    load_config_raw,
    resolve_config_path,
)
from cli.utils.output import console, print_error


def info(
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
    components: bool = typer.Option(
        False,
        "--components",
        "-c",
        help="Show detailed component information.",
    ),
) -> None:
    """
    Show detailed information about a configuration.

    For mode-aware configurations, displays modes, transition rules,
    and lifecycle hooks. For standard configurations, shows inputs,
    actions, and LLM settings.

    Examples
    --------
        om1 info spot
        om1 info spot_modes --components
        om1 info test --json
    """
    try:
        config_path = resolve_config_path(config_name)
        raw_config = load_config_raw(config_path)

        if json_output:
            import json

            console.print(json.dumps(raw_config, indent=2))
            return

        is_multi_mode = is_mode_aware_config(config_path)

        if is_multi_mode:
            _show_mode_aware_info(config_name, raw_config, components)
        else:
            _show_standard_info(config_name, raw_config, components)

    except FileNotFoundError:
        print_error(f"Configuration not found: {config_name}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        raise typer.Exit(1)


def _show_mode_aware_info(config_name: str, raw_config: dict, components: bool) -> None:
    """Display mode-aware configuration information."""
    # Lazy import to avoid loading heavy dependencies at CLI startup
    from runtime.multi_mode.config import load_mode_config

    try:
        mode_config = load_mode_config(config_name)

        # Header panel
        header = f"""[bold]Configuration:[/bold] {config_name}
[bold]Type:[/bold] Mode-Aware (Multi-Mode)
[bold]Default Mode:[/bold] {mode_config.default_mode}
[bold]Manual Switching:[/bold] {"Enabled" if mode_config.allow_manual_switching else "Disabled"}
[bold]Mode Memory:[/bold] {"Enabled" if mode_config.mode_memory_enabled else "Disabled"}"""

        if mode_config.global_lifecycle_hooks:
            header += f"\n[bold]Global Hooks:[/bold] {len(mode_config.global_lifecycle_hooks)}"

        console.print(Panel(header, title="Overview", border_style="blue"))

        # Modes table
        modes_table = Table(title="Available Modes", show_header=True)
        modes_table.add_column("Name", style="cyan")
        modes_table.add_column("Display Name", style="green")
        modes_table.add_column("Hertz", justify="right")
        modes_table.add_column("Inputs", justify="right")
        modes_table.add_column("Actions", justify="right")

        for name, mode in mode_config.modes.items():
            default_marker = " (DEFAULT)" if name == mode_config.default_mode else ""
            modes_table.add_row(
                name,
                f"{mode.display_name}{default_marker}",
                str(mode.hertz),
                str(len(mode._raw_inputs)),
                str(len(mode._raw_actions)),
            )

        console.print(modes_table)
        console.print()

        # Transition rules table
        if mode_config.transition_rules:
            rules_table = Table(title="Transition Rules", show_header=True)
            rules_table.add_column("From", style="yellow")
            rules_table.add_column("To", style="green")
            rules_table.add_column("Type")
            rules_table.add_column("Keywords")
            rules_table.add_column("Priority", justify="right")

            for rule in mode_config.transition_rules:
                from_display = (
                    mode_config.modes[rule.from_mode].display_name
                    if rule.from_mode != "*"
                    else "Any Mode"
                )
                to_display = mode_config.modes[rule.to_mode].display_name
                keywords = (
                    ", ".join(rule.trigger_keywords) if rule.trigger_keywords else "-"
                )

                rules_table.add_row(
                    from_display,
                    to_display,
                    rule.transition_type.value,
                    keywords[:30] + "..." if len(keywords) > 30 else keywords,
                    str(rule.priority),
                )

            console.print(rules_table)

        if components:
            _show_mode_components(mode_config)

    except Exception as e:
        print_error(f"Error loading mode configuration: {e}")
        raise typer.Exit(1)


def _show_standard_info(config_name: str, raw_config: dict, components: bool) -> None:
    """Display standard configuration information."""
    # Header panel
    header = f"""[bold]Configuration:[/bold] {config_name}
[bold]Type:[/bold] Standard (Single-Mode)
[bold]Name:[/bold] {raw_config.get("name", "N/A")}
[bold]Hertz:[/bold] {raw_config.get("hertz", "N/A")}
[bold]Version:[/bold] {raw_config.get("version", "N/A")}"""

    console.print(Panel(header, title="Overview", border_style="blue"))

    # Summary table
    summary_table = Table(title="Components Summary", show_header=True)
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Count", justify="right")

    summary_table.add_row("Inputs", str(len(raw_config.get("agent_inputs", []))))
    summary_table.add_row("Actions", str(len(raw_config.get("agent_actions", []))))
    summary_table.add_row("Simulators", str(len(raw_config.get("simulators", []))))
    summary_table.add_row("Backgrounds", str(len(raw_config.get("backgrounds", []))))

    console.print(summary_table)

    # LLM info
    if "cortex_llm" in raw_config:
        llm = raw_config["cortex_llm"]
        llm_info = f"""[bold]Type:[/bold] {llm.get("type", "N/A")}"""
        if "config" in llm:
            llm_config = llm["config"]
            if "agent_name" in llm_config:
                llm_info += f"\n[bold]Agent Name:[/bold] {llm_config['agent_name']}"
            if "history_length" in llm_config:
                llm_info += (
                    f"\n[bold]History Length:[/bold] {llm_config['history_length']}"
                )
        console.print(Panel(llm_info, title="LLM Configuration", border_style="green"))

    if components:
        _show_standard_components(raw_config)


def _show_mode_components(mode_config) -> None:
    """Show detailed components for mode-aware config."""
    for name, mode in mode_config.modes.items():
        console.print(f"\n[bold cyan]Mode: {name}[/bold cyan]")

        if mode._raw_inputs:
            inputs_table = Table(title="Inputs", show_header=True)
            inputs_table.add_column("Type", style="yellow")
            for inp in mode._raw_inputs:
                inputs_table.add_row(inp.get("type", "N/A"))
            console.print(inputs_table)

        if mode._raw_actions:
            actions_table = Table(title="Actions", show_header=True)
            actions_table.add_column("Name", style="green")
            actions_table.add_column("Label")
            actions_table.add_column("Connector")
            for action in mode._raw_actions:
                actions_table.add_row(
                    action.get("name", "N/A"),
                    action.get("llm_label", "N/A"),
                    action.get("connector", "N/A"),
                )
            console.print(actions_table)


def _show_standard_components(raw_config: dict) -> None:
    """Show detailed components for standard config."""
    console.print()

    if raw_config.get("agent_inputs"):
        inputs_table = Table(title="Inputs", show_header=True)
        inputs_table.add_column("Type", style="yellow")
        for inp in raw_config["agent_inputs"]:
            inputs_table.add_row(inp.get("type", "N/A"))
        console.print(inputs_table)

    if raw_config.get("agent_actions"):
        actions_table = Table(title="Actions", show_header=True)
        actions_table.add_column("Name", style="green")
        actions_table.add_column("Label")
        actions_table.add_column("Connector")
        for action in raw_config["agent_actions"]:
            actions_table.add_row(
                action.get("name", "N/A"),
                action.get("llm_label", "N/A"),
                action.get("connector", "N/A"),
            )
        console.print(actions_table)
