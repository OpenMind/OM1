"""Lint command - Run code quality checks."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

from cli.utils.config import get_src_dir
from cli.utils.output import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)


def lint(
    path: Optional[str] = typer.Argument(
        None,
        help="Specific file or directory to lint. Defaults to src/.",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        "-f",
        help="Automatically fix issues where possible.",
    ),
    check_only: bool = typer.Option(
        False,
        "--check",
        "-c",
        help="Only check, don't modify files (exit 1 if issues found).",
    ),
    format_code: bool = typer.Option(
        False,
        "--format",
        help="Also run code formatter (ruff format).",
    ),
    type_check: bool = typer.Option(
        False,
        "--type-check",
        "-t",
        help="Also run type checker (pyright).",
    ),
    all_checks: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Run all checks: lint, format, type-check.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output.",
    ),
) -> None:
    """
    Run code quality checks using ruff and optionally pyright.

    By default, runs ruff linter. Use --fix to auto-fix issues,
    --format to also format code, --type-check for type checking.

    Examples
    --------
        om1 lint                    # Check lint issues
        om1 lint --fix              # Fix lint issues
        om1 lint --format           # Lint + format
        om1 lint --all              # Lint + format + type-check
        om1 lint src/cli/           # Lint specific directory
        om1 lint --check            # Check mode (CI friendly)
    """
    project_root = get_src_dir().parent
    target_path = path or "src"

    target = Path(target_path)
    if not target.is_absolute():
        target = project_root / target_path

    if not target.exists():
        print_error(f"Path not found: {target_path}")
        raise typer.Exit(1)

    if all_checks:
        format_code = True
        type_check = True

    has_errors = False
    results: List[str] = []

    if _run_ruff_lint(str(target), fix, check_only, verbose, project_root):
        results.append("[green]lint: passed[/green]")
    else:
        results.append("[red]lint: failed[/red]")
        has_errors = True

    if format_code:
        if _run_ruff_format(
            str(target), fix or not check_only, check_only, verbose, project_root
        ):
            results.append("[green]format: passed[/green]")
        else:
            results.append("[red]format: failed[/red]")
            has_errors = True

    if type_check:
        if _run_pyright(str(target), verbose, project_root):
            results.append("[green]type-check: passed[/green]")
        else:
            results.append("[red]type-check: failed[/red]")
            has_errors = True

    console.print()
    console.print("[bold]Results:[/bold] " + " | ".join(results))

    if has_errors:
        if fix:
            print_warning("Some issues could not be auto-fixed.")
        else:
            print_info("Run with --fix to auto-fix issues.")
        raise typer.Exit(1)
    else:
        print_success("All checks passed!")


def _run_ruff_lint(
    target: str,
    fix: bool,
    check_only: bool,
    verbose: bool,
    cwd: Path,
) -> bool:
    """Run ruff linter."""
    console.print("[cyan]Running ruff lint...[/cyan]")

    cmd: List[str] = [sys.executable, "-m", "ruff", "check", target]

    if fix and not check_only:
        cmd.append("--fix")

    if verbose:
        cmd.append("--verbose")

    if verbose:
        print_info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=str(cwd))
        return result.returncode == 0
    except FileNotFoundError:
        print_error("ruff not found. Install with: uv add ruff")
        return False


def _run_ruff_format(
    target: str,
    apply: bool,
    check_only: bool,
    verbose: bool,
    cwd: Path,
) -> bool:
    """Run ruff formatter."""
    console.print("[cyan]Running ruff format...[/cyan]")

    cmd: List[str] = [sys.executable, "-m", "ruff", "format", target]

    if check_only:
        cmd.append("--check")

    if verbose:
        cmd.append("--verbose")

    if verbose:
        print_info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=str(cwd))
        return result.returncode == 0
    except FileNotFoundError:
        print_error("ruff not found. Install with: uv add ruff")
        return False


def _run_pyright(target: str, verbose: bool, cwd: Path) -> bool:
    """Run pyright type checker."""
    console.print("[cyan]Running pyright type-check...[/cyan]")

    cmd: List[str] = [sys.executable, "-m", "pyright", target]

    if verbose:
        cmd.append("--verbose")

    if verbose:
        print_info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=str(cwd))
        return result.returncode == 0
    except FileNotFoundError:
        print_warning("pyright not found. Install with: uv add pyright")
        return False
