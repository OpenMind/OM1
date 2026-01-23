"""Test command - Run tests with pytest."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

from cli.utils.config import get_src_dir
from cli.utils.output import console, print_error, print_info, print_success


def test(
    path: Optional[str] = typer.Argument(
        None,
        help="Specific test file or directory to run.",
    ),
    unit: bool = typer.Option(
        False,
        "--unit",
        "-u",
        help="Run only unit tests (tests/).",
    ),
    integration: bool = typer.Option(
        False,
        "--integration",
        "-i",
        help="Run only integration tests (tests_integration/).",
    ),
    all_tests: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Run all tests (unit + integration).",
    ),
    coverage: bool = typer.Option(
        False,
        "--coverage",
        "-c",
        help="Run with coverage report.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output.",
    ),
    markers: Optional[str] = typer.Option(
        None,
        "--markers",
        "-m",
        help="Run tests matching given mark expression (e.g., 'slow', 'not slow').",
    ),
    keyword: Optional[str] = typer.Option(
        None,
        "--keyword",
        "-k",
        help="Run tests matching given keyword expression.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        "-x",
        help="Stop on first failure.",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        "-p",
        help="Run tests in parallel (requires pytest-xdist).",
    ),
    last_failed: bool = typer.Option(
        False,
        "--last-failed",
        "--lf",
        help="Run only last failed tests.",
    ),
    lint: bool = typer.Option(
        False,
        "--lint",
        "-l",
        help="Run lint check before tests.",
    ),
) -> None:
    """
    Run tests using pytest.

    By default, runs unit tests. Use --integration for integration tests,
    or --all for both.

    Examples
    --------
        om1 test                     # Run unit tests
        om1 test --integration       # Run integration tests
        om1 test --all               # Run all tests
        om1 test --coverage          # Run with coverage
        om1 test --lint              # Run lint before tests
        om1 test tests/cli/          # Run specific directory
        om1 test -k "test_run"       # Run tests matching keyword
        om1 test -m slow             # Run tests with 'slow' marker
    """
    project_root = get_src_dir().parent

    # Run lint first if requested
    if lint:
        console.print("[cyan]Running lint check...[/cyan]\n")
        lint_cmd = [sys.executable, "-m", "ruff", "check", str(project_root / "src")]
        lint_result = subprocess.run(lint_cmd, cwd=str(project_root))
        if lint_result.returncode != 0:
            print_error("Lint check failed. Fix errors before running tests.")
            raise typer.Exit(lint_result.returncode)
        print_success("Lint check passed!")
        console.print()

    cmd: List[str] = [sys.executable, "-m", "pytest"]

    test_paths: List[str] = []

    if path:
        # Handle pytest selectors (e.g., tests/file.py::TestClass::test_method)
        if "::" in path:
            # Extract file path part before ::
            file_part = path.split("::")[0]
            test_file = Path(file_part)
            if not test_file.is_absolute():
                test_file = project_root / file_part
            if not test_file.exists():
                print_error(f"Test file not found: {file_part}")
                raise typer.Exit(1)
            # Pass full selector to pytest
            if not Path(path.split("::")[0]).is_absolute():
                test_paths.append(str(project_root / path))
            else:
                test_paths.append(path)
        else:
            test_path = Path(path)
            if not test_path.is_absolute():
                test_path = project_root / path
            if not test_path.exists():
                print_error(f"Test path not found: {path}")
                raise typer.Exit(1)
            test_paths.append(str(test_path))
    elif all_tests:
        unit_path = project_root / "tests"
        integration_path = project_root / "tests_integration"
        if unit_path.exists():
            test_paths.append(str(unit_path))
        if integration_path.exists():
            test_paths.append(str(integration_path))
    elif integration:
        integration_path = project_root / "tests_integration"
        if not integration_path.exists():
            print_error("Integration tests directory not found: tests_integration/")
            raise typer.Exit(1)
        test_paths.append(str(integration_path))
    elif unit:
        # Explicit unit test mode - only run tests/ directory
        unit_path = project_root / "tests"
        if not unit_path.exists():
            print_error("Unit tests directory not found: tests/")
            raise typer.Exit(1)
        test_paths.append(str(unit_path))
    else:
        # Default: run unit tests
        unit_path = project_root / "tests"
        if not unit_path.exists():
            print_error("Tests directory not found: tests/")
            raise typer.Exit(1)
        test_paths.append(str(unit_path))

    cmd.extend(test_paths)

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])

    if markers:
        cmd.extend(["-m", markers])

    if keyword:
        cmd.extend(["-k", keyword])

    if fail_fast:
        cmd.append("-x")

    if parallel:
        cmd.extend(["-n", "auto"])

    if last_failed:
        cmd.append("--lf")

    test_type = "all" if all_tests else ("integration" if integration else "unit")
    console.print(f"[cyan]Running {test_type} tests...[/cyan]\n")

    if verbose:
        print_info(f"Command: {' '.join(cmd)}")
        console.print()

    try:
        result = subprocess.run(cmd, cwd=str(project_root))

        if result.returncode == 0:
            print_success("All tests passed!")
        else:
            print_error(f"Tests failed with exit code {result.returncode}")
            raise typer.Exit(result.returncode)

    except FileNotFoundError:
        print_error("pytest not found. Install with: uv add pytest")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Test run interrupted.[/yellow]")
        raise typer.Exit(130)
