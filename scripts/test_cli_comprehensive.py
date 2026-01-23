#!/usr/bin/env python3
"""
Comprehensive CLI Parameter Test Script.

Tests all OM1 CLI commands and their parameters to find missing functionality.
Skips long-running commands like 'om1 run'.

Usage:
    uv run scripts/test_cli_comprehensive.py
"""

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestResult:
    """Result of a CLI test."""

    command: str
    success: bool
    exit_code: int
    output: str
    error: str
    notes: Optional[str] = None


def run_command(cmd: list[str], timeout: int = 30) -> TestResult:
    """Run a CLI command and return the result."""
    cmd_str = " ".join(cmd)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/Users/ganisacik/dapps/skillscld/OM1",
        )
        return TestResult(
            command=cmd_str,
            success=result.returncode == 0,
            exit_code=result.returncode,
            output=result.stdout[:500] if result.stdout else "",
            error=result.stderr[:500] if result.stderr else "",
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            command=cmd_str,
            success=False,
            exit_code=-1,
            output="",
            error="TIMEOUT",
            notes="Command timed out",
        )
    except Exception as e:
        return TestResult(
            command=cmd_str,
            success=False,
            exit_code=-1,
            output="",
            error=str(e),
            notes="Exception occurred",
        )


def test_help_commands() -> list[TestResult]:
    """Test all --help commands."""
    results = []
    commands = [
        ["uv", "run", "om1", "--help"],
        ["uv", "run", "om1", "run", "--help"],
        ["uv", "run", "om1", "stop", "--help"],
        ["uv", "run", "om1", "status", "--help"],
        ["uv", "run", "om1", "restart", "--help"],
        ["uv", "run", "om1", "list", "--help"],
        ["uv", "run", "om1", "info", "--help"],
        ["uv", "run", "om1", "validate", "--help"],
        ["uv", "run", "om1", "modes", "--help"],
        ["uv", "run", "om1", "init", "--help"],
        ["uv", "run", "om1", "doctor", "--help"],
        ["uv", "run", "om1", "setup", "--help"],
        ["uv", "run", "om1", "env", "--help"],
        ["uv", "run", "om1", "test", "--help"],
        ["uv", "run", "om1", "lint", "--help"],
    ]
    for cmd in commands:
        results.append(run_command(cmd))
    return results


def test_version() -> list[TestResult]:
    """Test version command."""
    return [
        run_command(["uv", "run", "om1", "--version"]),
        run_command(["uv", "run", "om1", "-V"]),
    ]


def test_list_command() -> list[TestResult]:
    """Test list command variations."""
    return [
        run_command(["uv", "run", "om1", "list"]),
    ]


def test_info_command() -> list[TestResult]:
    """Test info command variations."""
    return [
        run_command(["uv", "run", "om1", "info", "spot"]),
        run_command(["uv", "run", "om1", "info", "spot", "--json"]),
        run_command(["uv", "run", "om1", "info", "nonexistent_config"]),
        run_command(["uv", "run", "om1", "info", "spot_modes"]),  # mode-aware config
    ]


def test_validate_command() -> list[TestResult]:
    """Test validate command variations."""
    return [
        run_command(["uv", "run", "om1", "validate", "spot"]),
        run_command(["uv", "run", "om1", "validate", "spot", "--verbose"]),
        run_command(["uv", "run", "om1", "validate", "--all"]),
        run_command(["uv", "run", "om1", "validate", "--all", "--verbose"]),
        run_command(["uv", "run", "om1", "validate", "spot", "--skip-inputs"]),
        run_command(["uv", "run", "om1", "validate", "spot", "--allow-missing"]),
        run_command(["uv", "run", "om1", "validate", "nonexistent"]),
    ]


def test_modes_command() -> list[TestResult]:
    """Test modes command variations."""
    return [
        run_command(["uv", "run", "om1", "modes", "spot_modes"]),
        run_command(["uv", "run", "om1", "modes", "spot_modes", "--json"]),
        run_command(["uv", "run", "om1", "modes", "spot"]),  # non-mode-aware
        run_command(["uv", "run", "om1", "modes", "nonexistent"]),
    ]


def test_status_command() -> list[TestResult]:
    """Test status command variations."""
    results = [
        run_command(["uv", "run", "om1", "status"]),
        # --watch is tested with short timeout since it runs continuously
        run_command(["uv", "run", "om1", "status", "--watch"], timeout=3),
    ]
    return results


def test_stop_command() -> list[TestResult]:
    """Test stop command variations (safe - no agents running)."""
    return [
        run_command(["uv", "run", "om1", "stop", "nonexistent_agent"]),
        run_command(["uv", "run", "om1", "stop", "--all"]),
        run_command(["uv", "run", "om1", "stop", "test", "--force"]),
        run_command(["uv", "run", "om1", "stop", "test", "--timeout", "5"]),
    ]


def test_restart_command() -> list[TestResult]:
    """Test restart command variations (safe - no agents running)."""
    return [
        run_command(["uv", "run", "om1", "restart", "nonexistent"]),
        run_command(["uv", "run", "om1", "restart", "spot", "--hot-reload"]),
        run_command(["uv", "run", "om1", "restart", "spot", "--force"]),
    ]


def test_doctor_command() -> list[TestResult]:
    """Test doctor command."""
    return [
        run_command(["uv", "run", "om1", "doctor"]),
        run_command(["uv", "run", "om1", "doctor", "--config", "spot"]),
    ]


def test_env_command() -> list[TestResult]:
    """Test env command variations."""
    return [
        run_command(["uv", "run", "om1", "env", "list"]),
        run_command(["uv", "run", "om1", "env", "validate"]),
        run_command(["uv", "run", "om1", "env", "get", "OM_API_KEY"]),
        run_command(["uv", "run", "om1", "env", "generate"]),
    ]


def test_init_command() -> list[TestResult]:
    """Test init command (dry-run style tests)."""
    return [
        # Test with nonexistent template
        run_command(
            ["uv", "run", "om1", "init", "test_new", "--template", "nonexistent"]
        ),
    ]


def test_lint_command() -> list[TestResult]:
    """Test lint command variations."""
    return [
        run_command(["uv", "run", "om1", "lint", "--check", "src/cli/__init__.py"]),
        run_command(
            ["uv", "run", "om1", "lint", "--check", "--verbose", "src/cli/__init__.py"]
        ),
        run_command(
            ["uv", "run", "om1", "lint", "--check", "--format", "src/cli/__init__.py"]
        ),
        run_command(
            [
                "uv",
                "run",
                "om1",
                "lint",
                "--check",
                "--type-check",
                "src/cli/__init__.py",
            ]
        ),
        run_command(
            ["uv", "run", "om1", "lint", "--check", "--all", "src/cli/__init__.py"]
        ),
        run_command(["uv", "run", "om1", "lint", "nonexistent_path"]),
    ]


def test_test_command() -> list[TestResult]:
    """Test test command variations (run minimal tests)."""
    return [
        # Just check the command works, run a single fast test
        run_command(
            [
                "uv",
                "run",
                "om1",
                "test",
                "tests/cli/test_main.py::TestCLIMain::test_help",
                "-v",
            ]
        ),
        run_command(
            [
                "uv",
                "run",
                "om1",
                "test",
                "--unit",
                "tests/cli/test_main.py::TestCLIMain::test_help",
            ]
        ),
        run_command(["uv", "run", "om1", "test", "nonexistent_test_path"]),
        run_command(
            ["uv", "run", "om1", "test", "-k", "test_help", "tests/cli/test_main.py"]
        ),
        run_command(
            [
                "uv",
                "run",
                "om1",
                "test",
                "-x",
                "tests/cli/test_main.py::TestCLIMain::test_help",
            ]
        ),
    ]


def test_edge_cases() -> list[TestResult]:
    """Test edge cases and error handling."""
    return [
        # No arguments
        run_command(["uv", "run", "om1"]),
        # Invalid command
        run_command(["uv", "run", "om1", "invalid_command"]),
        # Missing required arguments
        run_command(["uv", "run", "om1", "info"]),
        run_command(["uv", "run", "om1", "validate"]),
    ]


def print_results(category: str, results: list[TestResult]) -> tuple[int, int]:
    """Print results for a category and return (passed, failed) counts."""
    print(f"\n{'='*60}")
    print(f" {category}")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for r in results:
        status = "PASS" if r.success else "FAIL"
        icon = "✓" if r.success else "✗"

        if r.success:
            passed += 1
            print(f"  {icon} [{status}] {r.command}")
        else:
            failed += 1
            print(f"  {icon} [{status}] {r.command}")
            print(f"      Exit code: {r.exit_code}")
            if r.error and r.error != "TIMEOUT":
                # Show first line of error
                error_line = r.error.split("\n")[0][:80]
                print(f"      Error: {error_line}")
            if r.notes:
                print(f"      Note: {r.notes}")

    return passed, failed


def main():
    """Run all CLI tests."""
    print("=" * 60)
    print(" OM1 CLI Comprehensive Parameter Test")
    print("=" * 60)
    print("\nRunning tests... (this may take a minute)\n")

    all_results = []
    total_passed = 0
    total_failed = 0

    # Run all test categories
    test_categories = [
        ("Help Commands", test_help_commands),
        ("Version", test_version),
        ("List Command", test_list_command),
        ("Info Command", test_info_command),
        ("Validate Command", test_validate_command),
        ("Modes Command", test_modes_command),
        ("Status Command", test_status_command),
        ("Stop Command", test_stop_command),
        ("Restart Command", test_restart_command),
        ("Doctor Command", test_doctor_command),
        ("Env Command", test_env_command),
        ("Init Command", test_init_command),
        ("Lint Command", test_lint_command),
        ("Test Command", test_test_command),
        ("Edge Cases", test_edge_cases),
    ]

    for category_name, test_func in test_categories:
        results = test_func()
        all_results.extend(results)
        passed, failed = print_results(category_name, results)
        total_passed += passed
        total_failed += failed

    # Summary
    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tests:  {total_passed + total_failed}")
    print(f"  Passed:       {total_passed}")
    print(f"  Failed:       {total_failed}")
    print(f"{'='*60}")

    # List all failures
    failures = [r for r in all_results if not r.success]
    if failures:
        print("\n FAILED TESTS:")
        print("-" * 60)
        for r in failures:
            print(f"  - {r.command}")
            if r.notes:
                print(f"    ({r.notes})")

    # Expected failures (commands that should fail)
    expected_failures = [
        "nonexistent",
        "invalid_command",
        "TIMEOUT",  # watch command timeout is expected
    ]

    unexpected_failures = [
        r
        for r in failures
        if not any(ef in r.command or ef in (r.notes or "") for ef in expected_failures)
    ]

    if unexpected_failures:
        print("\n UNEXPECTED FAILURES (bugs to investigate):")
        print("-" * 60)
        for r in unexpected_failures:
            print(f"  - {r.command}")
            print(
                f"    Exit: {r.exit_code}, Error: {r.error[:100] if r.error else 'None'}"
            )

    return 0 if not unexpected_failures else 1


if __name__ == "__main__":
    sys.exit(main())
