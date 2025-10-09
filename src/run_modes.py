import asyncio
import multiprocessing as mp
import os

import dotenv
import typer

from runtime.config import load_config
from runtime.cortex import CortexRuntime
from runtime.logging import setup_logging
from runtime.mode_aware_cortex import ModeAwareCortexRuntime
from runtime.mode_config import load_mode_config

app = typer.Typer()


@app.command()
def start(config_name: str, log_level: str = "INFO", log_to_file: bool = False) -> None:
    """Start the OM1 agent with a specific configuration."""
    setup_logging(config_name, log_level, log_to_file)

    # Try to load as mode-aware config first, fall back to regular config
    config_path = os.path.join(
        os.path.dirname(__file__), "../config", config_name + ".json5"
    )

    try:
        # Check if this is a mode configuration by looking for mode-specific keys
        import json5

        with open(config_path, "r") as f:
            raw_config = json5.load(f)

        if "modes" in raw_config and "default_mode" in raw_config:
            # This is a mode-aware configuration
            mode_config = load_mode_config(config_name)
            runtime = ModeAwareCortexRuntime(mode_config)
            print(f"Starting OM1 with mode-aware configuration: {config_name}")
            print(f"Available modes: {list(mode_config.modes.keys())}")
            print(f"Default mode: {mode_config.default_mode}")
        else:
            # This is a regular configuration
            config = load_config(config_name)
            runtime = CortexRuntime(config)
            print(f"Starting OM1 with standard configuration: {config_name}")

        # Start the runtime
        asyncio.run(runtime.run())

    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise typer.Exit(1)


@app.command()
def modes(config_name: str) -> None:
    """Show information about available modes in a mode-aware configuration."""
    try:
        mode_config = load_mode_config(config_name)

        print(f"Mode System: {mode_config.name}")
        print(f"Default Mode: {mode_config.default_mode}")
        print(
            f"Manual Switching: {'Enabled' if mode_config.allow_manual_switching else 'Disabled'}"
        )
        print(
            f"Announcements: {'Enabled' if mode_config.transition_announcement else 'Disabled'}"
        )
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
        print(f"Configuration file not found: {config_name}.json5")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error loading mode configuration: {e}")
        raise typer.Exit(1)


@app.command()
def list_configs() -> None:
    """List all available configuration files."""
    config_dir = os.path.join(os.path.dirname(__file__), "../config")

    if not os.path.exists(config_dir):
        print("Configuration directory not found")
        return

    configs = []
    mode_configs = []

    for filename in os.listdir(config_dir):
        if filename.endswith(".json5"):
            config_name = filename[:-6]  # Remove .json5 extension
            config_path = os.path.join(config_dir, filename)

            try:
                import json5

                with open(config_path, "r") as f:
                    raw_config = json5.load(f)

                if "modes" in raw_config and "default_mode" in raw_config:
                    mode_configs.append(
                        (config_name, raw_config.get("name", config_name))
                    )
                else:
                    configs.append((config_name, raw_config.get("name", config_name)))
            except:
                configs.append((config_name, "Invalid config"))

    if mode_configs:
        print("Mode-Aware Configurations:")
        print("-" * 30)
        for config_name, display_name in sorted(mode_configs):
            print(f"• {config_name} - {display_name}")
        print()

    if configs:
        print("Standard Configurations:")
        print("-" * 25)
        for config_name, display_name in sorted(configs):
            print(f"• {config_name} - {display_name}")


if __name__ == "__main__":

    # Fix for Linux multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    dotenv.load_dotenv()
    app()
