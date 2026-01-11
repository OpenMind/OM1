"""Validate command - Validate configuration files."""

import json
import os
import re
import traceback
from pathlib import Path
from typing import List, Tuple

import json5
import typer
from jsonschema import ValidationError, validate

from cli.utils.config import (
    get_config_dir,
    get_src_dir,
    list_configs,
    resolve_config_path,
)
from cli.utils.output import console, print_error, print_success, print_warning


def validate_cmd(
    config_name: str = typer.Argument(
        None,
        help="Configuration file name. Use --all to validate all configs.",
    ),
    all_configs: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Validate all configuration files.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation information.",
    ),
    check_components: bool = typer.Option(
        True,
        "--check-components/--no-check-components",
        help="Verify that all components exist in codebase.",
    ),
    skip_inputs: bool = typer.Option(
        False,
        "--skip-inputs",
        help="Skip input validation.",
    ),
    allow_missing: bool = typer.Option(
        False,
        "--allow-missing",
        help="Allow missing components (warn only).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on warnings too.",
    ),
) -> None:
    """
    Validate OM1 configuration files.

    Checks for valid JSON5 syntax, schema compliance, and component existence.

    Examples
    --------
        om1 validate spot
        om1 validate --all
        om1 validate test --verbose
        om1 validate spot --no-check-components
    """
    if all_configs:
        _validate_all_configs(
            verbose, check_components, skip_inputs, allow_missing, strict
        )
        return

    if config_name is None:
        print_error("Please provide a config name or use --all")
        raise typer.Exit(1)

    success = _validate_single_config(
        config_name, verbose, check_components, skip_inputs, allow_missing
    )

    if not success:
        raise typer.Exit(1)


def _validate_all_configs(
    verbose: bool,
    check_components: bool,
    skip_inputs: bool,
    allow_missing: bool,
    strict: bool,
) -> None:
    """Validate all configuration files."""
    standard_configs, mode_configs = list_configs()
    all_configs = [(name, "standard") for name, _ in standard_configs] + [
        (name, "mode-aware") for name, _ in mode_configs
    ]

    if not all_configs:
        print_warning("No configuration files found.")
        return

    console.print(f"Validating {len(all_configs)} configurations...\n")

    passed = 0
    failed = 0
    warnings = 0

    for config_name, config_type in all_configs:
        try:
            success = _validate_single_config(
                config_name,
                verbose=False,
                check_components=check_components,
                skip_inputs=skip_inputs,
                allow_missing=allow_missing,
                quiet=True,
            )
            if success:
                console.print(f"  [green]✓[/green] {config_name}")
                passed += 1
            else:
                console.print(f"  [red]✗[/red] {config_name}")
                failed += 1
        except Exception as e:
            console.print(f"  [red]✗[/red] {config_name}: {e}")
            failed += 1

    console.print()
    console.print(
        f"Results: [green]{passed} passed[/green], [red]{failed} failed[/red]"
    )

    if failed > 0 or (strict and warnings > 0):
        raise typer.Exit(1)


def _validate_single_config(
    config_name: str,
    verbose: bool,
    check_components: bool,
    skip_inputs: bool,
    allow_missing: bool,
    quiet: bool = False,
) -> bool:
    """Validate a single configuration file."""
    try:
        config_path = resolve_config_path(config_name)

        if verbose:
            console.print(f"Validating: {config_path}")
            console.print("-" * 50)

        # Load and parse JSON5
        with open(config_path, "r") as f:
            raw_config = json5.load(f)

        if verbose:
            print_success("JSON5 syntax valid")

        # Detect config type
        is_multi_mode = "modes" in raw_config and "default_mode" in raw_config
        config_type = "multi-mode" if is_multi_mode else "single-mode"

        if verbose:
            console.print(f"Detected {config_type} configuration")

        # Schema validation
        schema_file = (
            "multi_mode_schema.json" if is_multi_mode else "single_mode_schema.json"
        )
        schema_path = get_config_dir() / "schema" / schema_file

        with open(schema_path, "r") as f:
            schema = json.load(f)

        validate(instance=raw_config, schema=schema)

        if verbose:
            print_success("Schema validation passed")

        # Component validation
        if check_components:
            if not verbose and not quiet:
                console.print("Validating components...", end="")

            errors, warnings = _validate_components(
                raw_config, is_multi_mode, verbose, skip_inputs, allow_missing
            )

            if errors:
                if not verbose and not quiet:
                    console.print("\r" + " " * 40 + "\r", end="")
                for error in errors:
                    print_error(error)
                return False

            if warnings:
                for warning in warnings:
                    print_warning(warning)

            if not verbose and not quiet:
                console.print("\r" + " " * 40 + "\r", end="")

        # API key check
        _check_api_key(raw_config, verbose)

        if not quiet:
            console.print()
            print_success(f"Configuration '{config_name}' is valid!")

        return True

    except FileNotFoundError as e:
        if not quiet:
            print_error(str(e))
        return False

    except ValidationError as e:
        if not quiet:
            print_error("Schema validation failed")
            field_path = ".".join(str(p) for p in e.path) if e.path else "root"
            console.print(f"   Field: {field_path}")
            console.print(f"   Issue: {e.message}")
        return False

    except Exception as e:
        if not quiet:
            print_error(f"Validation error: {e}")
            if verbose:
                traceback.print_exc()
        return False


def _validate_components(
    raw_config: dict,
    is_multi_mode: bool,
    verbose: bool,
    skip_inputs: bool,
    allow_missing: bool,
) -> Tuple[List[str], List[str]]:
    """Validate that all components exist."""
    errors = []
    warnings = []

    if verbose:
        console.print("Checking component existence...")

    if is_multi_mode:
        # Global LLM
        if "cortex_llm" in raw_config:
            llm_type = raw_config["cortex_llm"].get("type")
            if llm_type and not _check_llm_exists(llm_type):
                msg = f"Global LLM type '{llm_type}' not found"
                if allow_missing:
                    warnings.append(msg)
                else:
                    errors.append(msg)

        # Validate each mode
        for mode_name, mode_data in raw_config.get("modes", {}).items():
            mode_errors, mode_warnings = _validate_mode_components(
                mode_name, mode_data, verbose, skip_inputs, allow_missing
            )
            errors.extend(mode_errors)
            warnings.extend(mode_warnings)
    else:
        mode_errors, mode_warnings = _validate_mode_components(
            "config", raw_config, verbose, skip_inputs, allow_missing
        )
        errors.extend(mode_errors)
        warnings.extend(mode_warnings)

    return errors, warnings


def _validate_mode_components(
    mode_name: str,
    mode_data: dict,
    verbose: bool,
    skip_inputs: bool,
    allow_missing: bool,
) -> Tuple[List[str], List[str]]:
    """Validate components for a single mode."""
    errors = []
    warnings = []

    # Inputs
    if not skip_inputs:
        for inp in mode_data.get("agent_inputs", []):
            input_type = inp.get("type")
            if input_type and not _check_input_exists(input_type):
                msg = f"[{mode_name}] Input type '{input_type}' not found"
                if allow_missing:
                    warnings.append(msg)
                else:
                    errors.append(msg)

    # LLM
    if "cortex_llm" in mode_data:
        llm_type = mode_data["cortex_llm"].get("type")
        if llm_type and not _check_llm_exists(llm_type):
            msg = f"[{mode_name}] LLM type '{llm_type}' not found"
            if allow_missing:
                warnings.append(msg)
            else:
                errors.append(msg)

    # Simulators
    for sim in mode_data.get("simulators", []):
        sim_type = sim.get("type")
        if sim_type and not _check_simulator_exists(sim_type):
            msg = f"[{mode_name}] Simulator type '{sim_type}' not found"
            if allow_missing:
                warnings.append(msg)
            else:
                errors.append(msg)

    # Actions
    for action in mode_data.get("agent_actions", []):
        action_name = action.get("name")
        if action_name and not _check_action_exists(action_name):
            msg = f"[{mode_name}] Action '{action_name}' not found"
            if allow_missing:
                warnings.append(msg)
            else:
                errors.append(msg)

    # Backgrounds
    for bg in mode_data.get("backgrounds", []):
        bg_type = bg.get("type")
        if bg_type and not _check_background_exists(bg_type):
            msg = f"[{mode_name}] Background type '{bg_type}' not found"
            if allow_missing:
                warnings.append(msg)
            else:
                errors.append(msg)

    return errors, warnings


def _check_input_exists(input_type: str) -> bool:
    """Check if input type exists."""
    plugins_dir = get_src_dir() / "inputs" / "plugins"
    return _check_class_exists(plugins_dir, input_type)


def _check_llm_exists(llm_type: str) -> bool:
    """Check if LLM type exists."""
    plugins_dir = get_src_dir() / "llm" / "plugins"
    return _check_class_exists(plugins_dir, llm_type)


def _check_simulator_exists(sim_type: str) -> bool:
    """Check if simulator type exists."""
    plugins_dir = get_src_dir() / "simulators" / "plugins"
    return _check_class_exists(plugins_dir, sim_type)


def _check_action_exists(action_name: str) -> bool:
    """Check if action exists."""
    interface_file = get_src_dir() / "actions" / action_name / "interface.py"
    return interface_file.exists()


def _check_background_exists(bg_type: str) -> bool:
    """Check if background type exists."""
    plugins_dir = get_src_dir() / "backgrounds" / "plugins"
    return _check_class_exists(plugins_dir, bg_type)


def _check_class_exists(plugins_dir: Path, class_name: str) -> bool:
    """Check if a class exists in plugin directory."""
    if not plugins_dir.exists():
        return False

    class_pattern = re.compile(rf"^class\s+{re.escape(class_name)}\s*\(", re.MULTILINE)

    for filepath in plugins_dir.glob("*.py"):
        if filepath.name == "__init__.py":
            continue
        try:
            content = filepath.read_text(encoding="utf-8")
            if class_pattern.search(content):
                return True
        except Exception:
            continue

    return False


def _check_api_key(raw_config: dict, verbose: bool) -> None:
    """Check API key configuration."""
    api_key = raw_config.get("api_key", "")
    env_api_key = os.environ.get("OM_API_KEY", "")

    if (not api_key or api_key == "openmind_free") and not env_api_key:
        print_warning("No API key configured")
        console.print("   Get a free key at: https://portal.openmind.org")
    elif verbose:
        if env_api_key:
            console.print("API key configured (from environment)")
        else:
            console.print("API key configured")
