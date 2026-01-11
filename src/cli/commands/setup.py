"""Setup command - Interactive setup wizard for OM1."""

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel

from cli.utils.config import list_configs
from cli.utils.output import console, print_error, print_info, print_success
from cli.utils.system import (
    CheckStatus,
    get_required_api_keys_for_config,
    get_required_extras_for_config,
    run_all_checks,
)


def setup(
    config_name: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Setup for specific config.",
    ),
    minimal: bool = typer.Option(
        False,
        "--minimal",
        help="Minimal setup for local LLM (Ollama) only.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Full setup with all features.",
    ),
    skip_deps: bool = typer.Option(
        False,
        "--skip-deps",
        help="Skip dependency installation.",
    ),
) -> None:
    """
    Interactive setup wizard for first-time users.

    Guides through dependency installation, environment configuration,
    and system validation.

    Examples
    --------
        om1 setup                    # Interactive setup
        om1 setup --minimal          # Quick setup for Ollama
        om1 setup --config twitter   # Setup for specific config
        om1 setup --full             # Install all dependencies
    """
    console.print(Panel("[bold]OM1 Setup Wizard[/bold]", border_style="blue"))
    console.print()

    # Step 1: Choose config if not specified
    if minimal:
        config_name = "ollama"
        console.print("[bold]Step 1:[/bold] Using minimal setup (Ollama)")
    elif full:
        config_name = None
        console.print("[bold]Step 1:[/bold] Full installation selected")
    elif config_name:
        console.print(
            f"[bold]Step 1:[/bold] Setting up for config: [cyan]{config_name}[/cyan]"
        )
    else:
        config_name = _choose_config()

    console.print()

    # Step 2: Install dependencies
    if not skip_deps:
        console.print("[bold]Step 2:[/bold] Installing dependencies...")
        _install_dependencies(config_name, full)
    else:
        console.print(
            "[bold]Step 2:[/bold] [dim]Skipping dependency installation[/dim]"
        )

    console.print()

    # Step 3: Setup environment
    console.print("[bold]Step 3:[/bold] Configuring environment...")
    _setup_environment(config_name)

    console.print()

    # Step 4: Validate
    console.print("[bold]Step 4:[/bold] Validating setup...")
    _validate_setup(config_name)

    console.print()

    # Summary
    console.print(
        Panel("[bold green]Setup Complete![/bold green]", border_style="green")
    )
    if config_name:
        console.print(f"\nTo start your agent: [cyan]om1 run {config_name}[/cyan]")
    else:
        console.print("\nTo start an agent: [cyan]om1 run <config>[/cyan]")
    console.print("To list configs:    [cyan]om1 list[/cyan]")
    console.print("To check status:    [cyan]om1 doctor[/cyan]")


def _choose_config() -> str:
    """Interactive config selection."""
    standard_configs, mode_configs = list_configs()

    # Combine and show options
    all_configs = [(name, "standard") for name, _ in standard_configs]
    all_configs += [(name, "mode-aware") for name, _ in mode_configs]

    # Recommended configs first
    recommended = ["ollama", "conversation", "open_ai"]
    recommended_found = []

    for name, _ in all_configs:
        if name in recommended:
            recommended_found.append(name)

    console.print("Choose a configuration to set up:\n")

    # Show recommended
    console.print("[bold]Recommended:[/bold]")
    for i, name in enumerate(recommended_found, 1):
        console.print(f"  {i}. [cyan]{name}[/cyan]")

    console.print("\n  0. See all configs")
    console.print()

    choice = typer.prompt("Enter number", default="1")

    try:
        choice_int = int(choice)
        if choice_int == 0:
            # Show all configs
            console.print("\n[bold]All Configs:[/bold]")
            for i, (name, _) in enumerate(all_configs, 1):
                console.print(f"  {i}. {name}")
            choice = typer.prompt("\nEnter number")
            choice_int = int(choice)
            return all_configs[choice_int - 1][0]
        elif 1 <= choice_int <= len(recommended_found):
            return recommended_found[choice_int - 1]
        else:
            print_error("Invalid choice")
            return "ollama"  # Default
    except (ValueError, IndexError):
        print_error("Invalid input, using ollama")
        return "ollama"


def _install_dependencies(config_name: Optional[str], full: bool) -> None:
    """Install required dependencies."""
    if full:
        extras = ["full"]
    elif config_name:
        extras = get_required_extras_for_config(config_name)
    else:
        extras = []

    if extras:
        extras_str = " ".join(f"--extra {e}" for e in extras)
        cmd = f"uv sync {extras_str}"
        console.print(f"  Running: [dim]{cmd}[/dim]")

        try:
            result = subprocess.run(
                ["uv", "sync"] + [arg for e in extras for arg in ["--extra", e]],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print_success("Dependencies installed")
            else:
                print_error(f"Failed to install dependencies: {result.stderr}")
        except subprocess.TimeoutExpired:
            print_error("Dependency installation timed out")
        except FileNotFoundError:
            print_error(
                "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
            )
    else:
        console.print("  [dim]No extra dependencies needed for this config[/dim]")
        print_success("Base dependencies already installed")


def _setup_environment(config_name: Optional[str]) -> None:
    """Setup environment variables."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    # Create .env.example if it doesn't exist
    if not env_example_path.exists():
        from cli.commands.env import ENV_TEMPLATE

        env_example_path.write_text(ENV_TEMPLATE)
        print_info("Created .env.example")

    # Create .env if it doesn't exist
    if not env_path.exists():
        env_path.write_text(env_example_path.read_text())
        print_info("Created .env from template")
        console.print("  [yellow]Please edit .env to add your API keys[/yellow]")
    else:
        print_info(".env already exists")

    # Show required API keys
    if config_name:
        api_keys = get_required_api_keys_for_config(config_name)
        if api_keys:
            console.print(f"\n  Required API keys for [cyan]{config_name}[/cyan]:")
            for key in api_keys:
                console.print(f"    - {key}")
        elif config_name == "ollama":
            console.print(
                "\n  [green]No API keys needed for Ollama (local LLM)[/green]"
            )
            console.print("  Make sure Ollama is running: [cyan]ollama serve[/cyan]")


def _validate_setup(config_name: Optional[str]) -> None:
    """Run validation checks."""
    results = run_all_checks(config_name)

    ok_count = sum(1 for r in results if r.status == CheckStatus.OK)
    warn_count = sum(1 for r in results if r.status == CheckStatus.WARN)
    fail_count = sum(1 for r in results if r.status == CheckStatus.FAIL)

    if fail_count == 0:
        print_success(f"All checks passed ({ok_count} OK, {warn_count} warnings)")
    else:
        print_error(f"{fail_count} checks failed")
        console.print("  Run [cyan]om1 doctor[/cyan] for details")
