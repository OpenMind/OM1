import ast
import json
import logging
import multiprocessing as mp
import os
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
    Categorizes into 'Mode-Aware' and 'Standard' configurations.
    """
    config_dir = os.path.join(os.path.dirname(__file__), "../config")

    if not os.path.exists(config_dir):
        logging.error(f"Configuration directory not found: {config_dir}")
        raise typer.Exit(1)

    configs = []
    mode_configs = []

    # Alfabetik sıralama için sorted() eklendi
    for filename in sorted(os.listdir(config_dir)):
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
            except Exception as e:
                # Geliştirilmiş hata bildirimi
                error_detail = f"Invalid config (Error: {str(e)})"
                configs.append((config_name, error_detail))
                logging.debug(f"Failed to load {filename}: {e}")

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
        help="Verify that all components exist in codebase",
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
    Validate an OM1 configuration file for syntax, schema and component existence.
    """
    try:
        config_path = _resolve_config_path(config_name)

        if verbose:
            print(f"Validating: {config_path}")
            print("-" * 50)

        try:
            with open(config_path, "r") as f:
                raw_config = json5.load(f)
        except ValueError as e:
            print("Error: Invalid JSON5 syntax")
            print(f"    {e}")
            raise typer.Exit(1)

        if verbose:
            print("JSON5 syntax valid")

        is_multi_mode = "modes" in raw_config and "default_mode" in raw_config
        config_type = "multi-mode" if is_multi_mode else "single-mode"

        if verbose:
            print(f"Detected {config_type} configuration")

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

        if check_components:
            if not verbose:
                print("Validating components...", end="", flush=True)
            _validate_components(
                raw_config, is_multi_mode, verbose, skip_inputs, allow_missing
            )
            if not verbose:
                print("\rAll components validated successfully!           ")

        _check_api_key(raw_config, verbose)

        print("\n" + "=" * 50)
        print("Configuration is valid!")
        print("=" * 50)

        if verbose:
            _print_config_summary(raw_config, is_multi_mode)

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found\n    {e}")
        raise typer.Exit(1)
    except ValidationError as e:
        field_path = ".".join(str(p) for p in e.path) if e.path else "root"
        print(f"Error: Schema validation failed\n    Field: {field_path}\n    Issue: {e.message}")
        raise typer.Exit(1)
    except Exception as e:
        if "Component validation" not in str(e):
            print(f"Error: Unexpected validation error\n    {e}")
            if verbose:
                traceback.print_exc()
        raise typer.Exit(1)


def _resolve_config_path(config_name: str) -> str:
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

    raise FileNotFoundError(f"Configuration '{config_name}' not found.")


def _validate_components(raw_config, is_multi_mode, verbose, skip_inputs=False, allow_missing=False):
    errors, warnings = [], []
    try:
        if is_multi_mode:
            if "cortex_llm" in raw_config:
                llm_type = raw_config["cortex_llm"].get("type")
                if llm_type and not _check_llm_exists(llm_type):
                    msg = f"Global LLM type '{llm_type}' not found"
                    if allow_missing: warnings.append(msg)
                    else: errors.append(msg)

            for mode_name, mode_data in raw_config.get("modes", {}).items():
                m_errors, m_warnings = _validate_mode_components(mode_name, mode_data, verbose, skip_inputs, allow_missing)
                errors.extend(m_errors); warnings.extend(m_warnings)
        else:
            m_errors, m_warnings = _validate_mode_components("config", raw_config, verbose, skip_inputs, allow_missing)
            errors.extend(m_errors); warnings.extend(m_warnings)
    except Exception as e:
        errors.append(f"Component validation error: {e}")

    if warnings:
        print("\nComponent validation warnings:")
        for w in warnings: print(f"   - {w}")
    if errors:
        print("\nComponent validation failed:")
        for e in errors: print(f"   - {e}")
        raise ValueError("Component validation failed")


def _validate_mode_components(mode_name, mode_data, verbose, skip_inputs=False, allow_missing=False):
    errors, warnings = [], []
    
    # Inputs
    if not skip_inputs:
        for inp in mode_data.get("agent_inputs", []):
            i_type = inp.get("type")
            if i_type and not _check_input_exists(i_type):
                msg = f"[{mode_name}] Input type '{i_type}' not found"
                if allow_missing: warnings.append(msg)
                else: errors.append(msg)
    
    # LLM
    if "cortex_llm" in mode_data:
        llm_type = mode_data["cortex_llm"].get("type")
        if llm_type and not _check_llm_exists(llm_type):
            msg = f"[{mode_name}] LLM type '{llm_type}' not found"
            if allow_missing: warnings.append(msg)
            else: errors.append(msg)

    # Actions, Simulators, Backgrounds
    checks = [
        ("agent_actions", "name", _check_action_exists, "Action"),
        ("simulators", "type", _check_simulator_exists, "Simulator"),
        ("backgrounds", "type", _check_background_exists, "Background")
    ]
    
    for key, field, func, label in checks:
        for item in mode_data.get(key, []):
            val = item.get(field)
            if val and not func(val):
                msg = f"[{mode_name}] {label} '{val}' not found"
                if allow_missing: warnings.append(msg)
                else: errors.append(msg)
                
    return errors, warnings


def _check_class_in_dir(directory: str, class_name: str) -> bool:
    if not os.path.exists(directory): return False
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef) and node.name == class_name:
                            return True
            except: continue
    return False


def _check_input_exists(t): return _check_class_in_dir(os.path.join(os.path.dirname(__file__), "inputs", "plugins"), t)
def _check_llm_exists(t): return _check_class_in_dir(os.path.join(os.path.dirname(__file__), "llm", "plugins"), t)
def _check_simulator_exists(t): return _check_class_in_dir(os.path.join(os.path.dirname(__file__), "simulators", "plugins"), t)
def _check_background_exists(t): return _check_class_in_dir(os.path.join(os.path.dirname(__file__), "backgrounds", "plugins"), t)
def _check_action_exists(n): return os.path.exists(os.path.join(os.path.dirname(__file__), "actions", n, "interface.py"))


def _check_api_key(raw_config, verbose):
    api_key = raw_config.get("api_key", "")
    env_key = os.environ.get("OM_API_KEY", "")
    if (not api_key or api_key == "openmind_free") and not env_key:
        print("\nWarning: No API key configured. Get one at: https://portal.openmind.org")
    elif verbose:
        print("API key configured" + (" (from env)" if env_key else ""))


def _print_config_summary(raw_config, is_multi_mode):
    print(f"\nConfiguration Summary:\n{'-' * 50}")
    if is_multi_mode:
        print(f"   Type: Multi-mode\n   Name: {raw_config.get('name', 'N/A')}\n   Modes: {len(raw_config.get('modes', {}))}")
    else:
        print(f"   Type: Single-mode\n   Name: {raw_config.get('name', 'N/A')}\n   Inputs: {len(raw_config.get('agent_inputs', []))}")


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")
    dotenv.load_dotenv()
    app()
