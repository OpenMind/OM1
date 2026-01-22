"""
OM1 CLI - Main entry point.

Usage:
    om1 run <config>        Run an agent with configuration
    om1 stop [config]       Stop running agent(s)
    om1 status              Show running agents
    om1 restart <config>    Restart an agent
    om1 list                List available configurations
    om1 info <config>       Show configuration details
    om1 validate <config>   Validate a configuration
    om1 doctor              Check system health
    om1 setup               Interactive setup wizard
    om1 env <action>        Manage environment variables
    om1 test                Run tests
    om1 lint                Run linters
    om1 init <name>         Create a new configuration
"""

import multiprocessing as mp
from typing import Optional

import dotenv
import typer

# Bootstrap must be imported first to configure sys.path for zenoh_msgs conflict
import cli._bootstrap  # noqa: F401
from cli import __version__
from cli.commands.doctor import doctor
from cli.commands.env import env
from cli.commands.info import info
from cli.commands.init import init
from cli.commands.lint import lint
from cli.commands.list_cmd import list_configs
from cli.commands.modes import modes
from cli.commands.restart import restart
from cli.commands.run import run
from cli.commands.setup import setup
from cli.commands.status import status
from cli.commands.stop import stop
from cli.commands.test import test
from cli.commands.validate import validate_cmd
from cli.utils.output import console

app = typer.Typer(
    name="om1",
    help="OM1 - OpenMind AI Runtime CLI",
    add_completion=True,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Display version and exit."""
    if value:
        console.print(f"OM1 CLI version: [cyan]{__version__}[/cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """
    OM1 - OpenMind AI Runtime CLI.

    A modular AI runtime that empowers developers to create and deploy
    multimodal AI agents across digital environments and physical robots.
    """
    pass


# Agent lifecycle commands
app.command(name="run")(run)
app.command(name="stop")(stop)
app.command(name="status")(status)
app.command(name="restart")(restart)

# Configuration commands
app.command(name="list")(list_configs)
app.command(name="info")(info)
app.command(name="validate")(validate_cmd)
app.command(name="modes")(modes)
app.command(name="init")(init)

# System commands
app.command(name="doctor")(doctor)
app.command(name="setup")(setup)
app.command(name="env")(env)

# Development commands
app.command(name="test")(test)
app.command(name="lint")(lint)


def cli_main() -> None:
    """Main entry point for CLI."""
    # Fix for Linux multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    dotenv.load_dotenv()
    app()


if __name__ == "__main__":
    cli_main()
