import json
import logging
import multiprocessing as mp
import os
import re
import traceback

import dotenv
import json5
import typer
from jsonschema import ValidationError, validate

from runtime.multi_mode.config import load_mode_config

app = typer.Typer()


@app.command()
def modes(config_name: str) -> None:
    """
    Show detailed information about available modes, transition rules,
    and settings within a specified mode-aware configuration file.

    This command is crucial for debugging and understanding the current
    state of the multi-mode system configuration.

    Parameters
    ----------
    config_name : str
        The name of the configuration file (e.g., 'example' for 'example.json5')
        located in the '../config' directory.

    Raises
    ------
    typer.Exit(1)
        If the configuration file is not found or fails to load.
    """
    try:
        mode_config = load_mode_config(config_name)

        print("-" * 32)
        print(f"Mode System: {mode_config.name}")
        print(f"Default Mode: {mode_config.default_mode}")
        print(
            f"Manual Switching: {'Enabled' if mode_config.allow_manual_switching else 'Disabled'}"
        )
        print(
            f"Mode Memory: {'Enabled' if mode_config.mode_memory_enabled else 'Disabled'}"
        )

        if mode_config.global_lifecycle_hooks:
            print(f"Global Lifecycle Hooks: {len(mode_config.global_lifecycle_hooks)}")
        print()

        print("Available Modes:")
        print("-" * 50)
        for name, mode in mode_config.modes.items():
            is_default = " (DEFAULT)" if name == mode_config.default_mode else ""
            print(f"• {mode.display_name}{is_default}")
            print(f"  Name: {name}")
            print(f"  Description: {mode.description}")
            print(f"  Frequency: {mode.hertz} Hz")
            if mode.timeout_seconds:
                print(f"  Timeout: {mode.timeout_seconds} seconds")
            print(f"  Inputs: {len(mode._raw_inputs)}")
            print(f"  Actions: {len(mode._raw_actions)}")
            if mode.lifecycle_hooks:
                print(f"  Lifecycle Hooks: {len(mode.lifecycle_hooks)}")
            print()

        print("Transition Rules:")
        print("-" * 50)
        for rule in mode_config.transition_rules:
            from_display = (
                mode_config.modes[rule.from_mode].display_name
                if rule.from_mode != "*"
                else "Any Mode"
            )
            to_display = mode_config.modes[rule.to_mode].display_name
            print(f"• {from_display} → {to_display}")
            print(f"  Type: {rule.transition_type.value}")
            if rule.trigger_keywords:
                print(f"  Keywords: {', '.join(rule.trigger_keywords)}")
            print(f"  Priority: {rule.priority}")
            if rule.cooldown_seconds > 0:
                print(f"  Cooldown: {rule.cooldown_seconds}s")
            print()

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_name}.json5")
        raise typer.Exit(1)
    except Exception as e:
        logging.error(f"Error loading mode configuration: {e}")
        raise typer.Exit(1)


@app.command()
def list_configs() -> None:
    """
    List all available configuration files found in the '../config' directory.

    It categorizes the files into 'Mode-Aware Configurations' (those containing
    'modes' and 'default_mode' keys) and 'Standard Configurations' (all others).
    This helps the user quickly identify configurations for the multi-mode runtime.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "../config")

    if not os.path.exists(config_dir):
        print("Configuration directory not found")
        return

    configs = []
    mode_configs = []

    for filename in os.listdir(config_dir):
        if filename.endswith(".json5"):
            config_name = filename[:-6]
            config_path = os.path.join(config_dir, filename)

            try:
                with open(config_path, "r") as f:
                    raw_config = json5.load(f)

                if "modes" in raw_config and "default_mode" in raw_config:
                    mode_configs.append(
                        (config_name, raw_config.get("name", config_name))
                    )
                else:
                    configs.append((config_name, raw_config.get("name", config_name)))
            except Exception as _:
                configs.append((config_name, "Invalid config"))

    print("-" * 32)
    if mode_configs:
        print("Mode-Aware Configurations:")
        print("-" * 32)
        for config_name, display_name in sorted(mode_configs):
            print(f"• {config_name} - {display_name}")
        print()

    print("-" * 32)
    if configs:
        print("Standard Configurations:")
        print("-" * 32)
        for config_name, display_name in sorted(configs):
            print(f"• {config_name} - {display_name}")


@app.command()
def validate_config(
    config_name: str = typer.Argument(
        ...,
        help="Configuration file name or path (e.g., 'test' or 'config/test.json5')",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed validation information"
    ),
    check_components: bool = typer.Option(
        True,
        "--check-components",
        help="Verify that all components (inputs, LLMs, actions) exist in codebase (slower but thorough)",
    ),
    skip_inputs: bool = typer.Option(
        False,
        "--skip-inputs",
        help="Skip input validation (useful for debugging)",
    ),
    allow_missing: bool = typer.Option(
        False,
        "--allow-missing",
        help="Allow missing components (only warn, don't fail)",
    ),
) -> None:
    """
    Validate an OM1 configuration file.

    Checks for:
    - Valid JSON5 syntax
    - Schema compliance (required fields, correct types)
    - API key configuration (warning only)
    - Component existence (with --check-components flag)

    Examples
    --------
        uv run src/cli.py validate-config test
        uv run src/cli.py validate-config config/my_robot.json5
        uv run src/cli.py validate-config test --verbose
        uv run src/cli.py validate-config test --check-components
        uv run src/cli.py validate-config test --check-components --skip-inputs
        uv run src/cli.py validate-config test --check-components --allow-missing
    """
    try:
        # Resolve config path
        config_path = _resolve_config_path(config_name)

        if verbose:
            print(f"Validating: {config_path}")
            print("-" * 50)

        # Load and parse JSON5
        try:
            with open(config_path, "r") as f:
                raw_config = json5.load(f)
        except ValueError as e:
            print("Error: Invalid JSON5 syntax")
            print(f"   {e}")
            raise typer.Exit(1)

        if verbose:
            print("JSON5 syntax valid")

        # Detect config type
        is_multi_mode = "modes" in raw_config and "default_mode" in raw_config
        config_type = "multi-mode" if is_multi_mode else "single-mode"

        if verbose:
            print(f"Detected {config_type} configuration")

        # Schema validation
        schema_file = (
            "multi_mode_schema.json" if is_multi_mode else "single_mode_schema.json"
        )
        schema_path = os.path.join(
            os.path.dirname(__file__), "../config/schema", schema_file
        )

        with open(schema_path, "r") as f:
            schema = json.load(f)

        validate(instance=raw_config, schema=schema)

        if verbose:
            print("Schema validation passed")

        # Component validation (if requested)
        if check_components:
            if not verbose:
                print(
                    "Validating components (this may take a moment)...",
                    end="",
                    flush=True,
                )
            _validate_components(
                raw_config, is_multi_mode, verbose, skip_inputs, allow_missing
            )
            if not verbose:
                print("\rAll components validated successfully!           ")

        # API key check (warning only)
        _check_api_key(raw_config, verbose)

        # Success message
        print()
        print("=" * 50)
        print("Configuration is valid!")
        print("=" * 50)

        if verbose:
            _print_config_summary(raw_config, is_multi_mode)

    except FileNotFoundError as e:
        print("Error: Configuration file not found")
        print(f"   {e}")
        raise typer.Exit(1)

    except ValueError as e:
        if "Component validation" in str(e):
            pass  # Already printed by _validate_components
        else:
            print("Error: Unexpected validation error")
            print(f"   {e}")
            if verbose:

                traceback.print_exc()
        raise typer.Exit(1)

    except ValidationError as e:
        print("Error: Schema validation failed")
        field_path = ".".join(str(p) for p in e.path) if e.path else "root"
        print(f"   Field: {field_path}")
        print(f"   Issue: {e.message}")
        if verbose and e.schema:
            print("\n   Schema requirement:")
            print(f"   {e.schema}")
        raise typer.Exit(1)

    except Exception as e:
        if "Component validation" not in str(e):
            print("Error: Unexpected validation error")
            print(f"   {e}")
            if verbose:

                traceback.print_exc()
        raise typer.Exit(1)


def _resolve_config_path(config_name: str) -> str:
    """
    Resolve configuration path from name or path.

    Parameters
    ----------
    config_name : str
        Configuration name or path

    Returns
    -------
    str
        Absolute path to configuration file

    Raises
    ------
    FileNotFoundError
        If configuration file cannot be found
    """
    if os.path.exists(config_name):
        return os.path.abspath(config_name)

    if os.path.exists(config_name + ".json5"):
        return os.path.abspath(config_name + ".json5")

    config_dir = os.path.join(os.path.dirname(__file__), "../config")
    config_path = os.path.join(config_dir, config_name)

    if os.path.exists(config_path):
        return os.path.abspath(config_path)

    if os.path.exists(config_path + ".json5"):
        return os.path.abspath(config_path + ".json5")

    raise FileNotFoundError(
        f"Configuration '{config_name}' not found. "
        f"Tried: {config_name}, {config_name}.json5, {config_path}, {config_path}.json5"
    )


def _validate_components(
    raw_config: dict,
    is_multi_mode: bool,
    verbose: bool,
    skip_inputs: bool = False,
    allow_missing: bool = False,
):
    """
    Validate that all component types exist in codebase.

    Parameters
    ----------
    raw_config : dict
        Raw configuration dictionary
    is_multi_mode : bool
        Whether this is a multi-mode configuration
    verbose : bool
        Whether to print verbose output
    skip_inputs : bool
        Whether to skip input validation
    allow_missing : bool
        Whether to allow missing components (warnings only)

    Raises
    ------
    ValueError
        If component validation fails and allow_missing is False
    """
    errors = []
    warnings = []

    if verbose:
        print("Checking component existence...")

    try:
        if is_multi_mode:
            if "cortex_llm" in raw_config:
                llm_type = raw_config["cortex_llm"].get("type")
                if llm_type and verbose:
                    print(f"  Checking global LLM: {llm_type}")
                if llm_type and not _check_llm_exists(llm_type):
                    msg = f"Global LLM type '{llm_type}' not found"
                    if allow_missing:
                        warnings.append(msg)
                    else:
                        errors.append(msg)

            for mode_name, mode_data in raw_config.get("modes", {}).items():
                if verbose:
                    print(f"  Validating mode: {mode_name}")
                mode_errors, mode_warnings = _validate_mode_components(
                    mode_name, mode_data, verbose, skip_inputs, allow_missing
                )
                errors.extend(mode_errors)
                warnings.extend(mode_warnings)
        else:
            if verbose:
                print("  Validating single-mode configuration")
            mode_errors, mode_warnings = _validate_mode_components(
                "config", raw_config, verbose, skip_inputs, allow_missing
            )
            errors.extend(mode_errors)
            warnings.extend(mode_warnings)

    except Exception as e:
        error_msg = f"Component validation error: {e}"
        if allow_missing:
            warnings.append(error_msg)
        else:
            errors.append(error_msg)
        if verbose:

            traceback.print_exc()

    if warnings:
        print("Component validation warnings:")
        for warning in warnings:
            print(f"   - {warning}")

    if errors:
        print("Component validation failed:")
        for error in errors:
            print(f"   - {error}")
        raise ValueError("Component validation failed")

    if verbose:
        print("All components exist")


def _validate_mode_components(
    mode_name: str,
    mode_data: dict,
    verbose: bool,
    skip_inputs: bool = False,
    allow_missing: bool = False,
) -> tuple:
    """
    Validate components for a single mode.

    Parameters
    ----------
    mode_name : str
        Name of the mode being validated
    mode_data : dict
        Mode configuration data
    verbose : bool
        Whether to print verbose output
    skip_inputs : bool
        Whether to skip input validation
    allow_missing : bool
        Whether to allow missing components

    Returns
    -------
    tuple
        (errors, warnings) lists
    """
    errors = []
    warnings = []

    try:
        if not skip_inputs:
            inputs = mode_data.get("agent_inputs", [])
            if verbose and inputs:
                print(f"    Checking {len(inputs)} inputs...")

            for inp in inputs:
                input_type = inp.get("type")
                if input_type:
                    if verbose:
                        print(f"      Input: {input_type}", end=" ")
                    if not _check_input_exists(input_type):
                        msg = f"[{mode_name}] Input type '{input_type}' not found"
                        if allow_missing:
                            warnings.append(msg)
                            if verbose:
                                print("(warning)")
                        else:
                            errors.append(msg)
                            if verbose:
                                print("(not found)")
                    else:
                        if verbose:
                            print("OK")
        else:
            if verbose:
                print("    Skipping input validation")

        if "cortex_llm" in mode_data:
            llm_type = mode_data["cortex_llm"].get("type")
            if llm_type:
                if verbose:
                    print(f"    LLM: {llm_type}", end=" ")
                if not _check_llm_exists(llm_type):
                    msg = f"[{mode_name}] LLM type '{llm_type}' not found"
                    if allow_missing:
                        warnings.append(msg)
                        if verbose:
                            print("(warning)")
                    else:
                        errors.append(msg)
                        if verbose:
                            print("(not found)")
                else:
                    if verbose:
                        print("OK")

        simulators = mode_data.get("simulators", [])
        if verbose and simulators:
            print(f"    Checking {len(simulators)} simulators...")

        for sim in simulators:
            sim_type = sim.get("type")
            if sim_type:
                if verbose:
                    print(f"      Simulator: {sim_type}", end=" ")
                if not _check_simulator_exists(sim_type):
                    msg = f"[{mode_name}] Simulator type '{sim_type}' not found"
                    if allow_missing:
                        warnings.append(msg)
                        if verbose:
                            print("(warning)")
                    else:
                        errors.append(msg)
                        if verbose:
                            print("(not found)")
                else:
                    if verbose:
                        print("OK")

        actions = mode_data.get("agent_actions", [])
        if verbose and actions:
            print(f"    Checking {len(actions)} actions...")

        for action in actions:
            action_name = action.get("name")
            if action_name:
                if verbose:
                    print(f"      Action: {action_name}", end=" ")
                if not _check_action_exists(action_name):
                    msg = f"[{mode_name}] Action '{action_name}' not found"
                    if allow_missing:
                        warnings.append(msg)
                        if verbose:
                            print("(warning)")
                    else:
                        errors.append(msg)
                        if verbose:
                            print("(not found)")
                else:
                    if verbose:
                        print("OK")

        backgrounds = mode_data.get("backgrounds", [])
        if verbose and backgrounds:
            print(f"    Checking {len(backgrounds)} backgrounds...")

        for bg in backgrounds:
            bg_type = bg.get("type")
            if bg_type:
                if verbose:
                    print(f"      Background: {bg_type}", end=" ")
                if not _check_background_exists(bg_type):
                    msg = f"[{mode_name}] Background type '{bg_type}' not found"
                    if allow_missing:
                        warnings.append(msg)
                        if verbose:
                            print("(warning)")
                    else:
                        errors.append(msg)
                        if verbose:
                            print("(not found)")
                else:
                    if verbose:
                        print("OK")

    except Exception as e:
        msg = f"[{mode_name}] Error during validation: {e}"
        if allow_missing:
            warnings.append(msg)
        else:
            errors.append(msg)
        if verbose:
            print(f"    Error: {e}")

    return errors, warnings


def _check_input_exists(input_type: str) -> bool:
    """
    Check if input type exists by searching for class definition in plugin files.

    Parameters
    ----------
    input_type : str
        Input type name to check

    Returns
    -------
    bool
        True if input type exists, False otherwise
    """
    src_dir = os.path.dirname(__file__)
    plugins_dir = os.path.join(src_dir, "inputs", "plugins")

    if not os.path.exists(plugins_dir):
        return False

    # Search for class definition in all .py files
    class_pattern = re.compile(rf"^class\s+{re.escape(input_type)}\s*\(", re.MULTILINE)

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(plugins_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if class_pattern.search(content):
                        return True
            except Exception:
                continue

    return False


def _check_llm_exists(llm_type: str) -> bool:
    """
    Check if LLM type exists by searching for class definition in plugin files.

    Parameters
    ----------
    llm_type : str
        LLM type name to check

    Returns
    -------
    bool
        True if LLM type exists, False otherwise
    """
    src_dir = os.path.dirname(__file__)
    plugins_dir = os.path.join(src_dir, "llm", "plugins")

    if not os.path.exists(plugins_dir):
        return False

    # Search for class definition in all .py files
    class_pattern = re.compile(rf"^class\s+{re.escape(llm_type)}\s*\(", re.MULTILINE)

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(plugins_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if class_pattern.search(content):
                        return True
            except Exception:
                continue

    return False


def _check_simulator_exists(sim_type: str) -> bool:
    """
    Check if simulator type exists by searching for class definition in plugin files.

    Parameters
    ----------
    sim_type : str
        Simulator type name to check

    Returns
    -------
    bool
        True if simulator type exists, False otherwise
    """
    src_dir = os.path.dirname(__file__)
    plugins_dir = os.path.join(src_dir, "simulators", "plugins")

    if not os.path.exists(plugins_dir):
        return False

    # Search for class definition in all .py files
    class_pattern = re.compile(rf"^class\s+{re.escape(sim_type)}\s*\(", re.MULTILINE)

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(plugins_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if class_pattern.search(content):
                        return True
            except Exception:
                continue

    return False


def _check_action_exists(action_name: str) -> bool:
    """
    Check if action exists by verifying interface file presence.

    Parameters
    ----------
    action_name : str
        Action name to check

    Returns
    -------
    bool
        True if action exists, False otherwise
    """
    src_dir = os.path.dirname(__file__)
    interface_file = os.path.join(src_dir, "actions", action_name, "interface.py")
    return os.path.exists(interface_file)


def _check_background_exists(bg_type: str) -> bool:
    """
    Check if background type exists by searching for class definition in plugin files.

    Parameters
    ----------
    bg_type : str
        Background type name to check

    Returns
    -------
    bool
        True if background type exists, False otherwise
    """
    src_dir = os.path.dirname(__file__)
    plugins_dir = os.path.join(src_dir, "backgrounds", "plugins")

    if not os.path.exists(plugins_dir):
        return False

    # Search for class definition in all .py files
    class_pattern = re.compile(rf"^class\s+{re.escape(bg_type)}\s*\(", re.MULTILINE)

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(plugins_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if class_pattern.search(content):
                        return True
            except Exception:
                continue

    return False


def _check_api_key(raw_config: dict, verbose: bool):
    """
    Check API key configuration (warning only).

    Parameters
    ----------
    raw_config : dict
        Raw configuration dictionary
    verbose : bool
        Whether to print verbose output
    """
    api_key = raw_config.get("api_key", "")
    env_api_key = os.environ.get("OM_API_KEY", "")

    if (not api_key or api_key == "openmind_free") and not env_api_key:
        print()
        print("Warning: No API key configured")
        print("   Get a free key at: https://portal.openmind.org")
        print("   Or set OM_API_KEY in your .env file")
    elif verbose:
        if env_api_key:
            print("API key configured (from environment)")
        else:
            print("API key configured")


def _print_config_summary(raw_config: dict, is_multi_mode: bool):
    """
    Print configuration summary.

    Parameters
    ----------
    raw_config : dict
        Raw configuration dictionary
    is_multi_mode : bool
        Whether this is is a multi multi-mode configuration
    """
    print()
    print("Configuration Summary:")
    print("-" * 50)

    if is_multi_mode:
        print("   Type: Multi-mode")
        print(f"   Name: {raw_config.get('name', 'N/A')}")
        print(f"   Default Mode: {raw_config.get('default_mode')}")
        print(f"   Modes: {len(raw_config.get('modes', {}))}")
        print(f"   Transition Rules: {len(raw_config.get('transition_rules', []))}")
    else:
        print("   Type: Single-mode")
        print(f"   Name: {raw_config.get('name', 'N/A')}")
        print(f"   Frequency: {raw_config.get('hertz', 'N/A')} Hz")
        print(f"   Inputs: {len(raw_config.get('agent_inputs', []))}")
        print(f"   Actions: {len(raw_config.get('agent_actions', []))}")


if __name__ == "__main__":

    # Fix for Linux multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    dotenv.load_dotenv()
    app()
