import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import shutil
import sys
from typing import Optional, Tuple

import dotenv
import json5

from runtime.logging import setup_logging
from runtime.multi_mode.config import load_mode_config
from runtime.multi_mode.cortex import ModeCortexRuntime
from runtime.single_mode.config import load_config
from runtime.single_mode.cortex import CortexRuntime


def setup_config_file(config_name: Optional[str]) -> Tuple[str, str]:
    """
    Set up the configuration file.

    Parameters
    ----------
    config_name : str, optional
        The name of the configuration file (without extension) located in the config directory.
        If not provided, uses .runtime.json5 from memory folder.
    """
    # If no config_name is provided, use the default .runtime.json5 from memory
    if config_name is None:
        runtime_config_path = os.path.join(
            os.path.dirname(__file__), "../config/memory", ".runtime.json5"
        )

        if not os.path.exists(runtime_config_path):
            logging.error(
                f"Default runtime configuration file not found: {runtime_config_path}"
            )
            logging.error(
                "Please provide a config_name or ensure .runtime.json5 exists in config/memory/"
            )
            sys.exit(1)

        config_name = ".runtime"
        config_path = os.path.join(
            os.path.dirname(__file__), "../config", config_name + ".json5"
        )

        shutil.copy2(runtime_config_path, config_path)
        logging.info("Using default runtime configuration from memory folder")
        logging.info(
            f"Copied config/memory/.runtime.json5 to config/{config_name}.json5 for system compatibility"
        )
    else:
        config_path = os.path.join(
            os.path.dirname(__file__), "../config", config_name + ".json5"
        )

    return config_name, config_path


def start(
    config_name: Optional[str],
    hot_reload: bool,
    check_interval: int,
    log_level: str,
    log_to_file: bool,
) -> None:
    """
    Start the OM1 agent with a specific configuration.

    Parameters
    ----------
    config_name : str, optional
        The name of the configuration file (without extension) located in the config directory.
        If not provided, uses .runtime.json5 from memory folder as default.
    hot_reload : bool, optional
        Enable hot-reload of configuration files (default is True).
    check_interval : int, optional
        Interval in seconds between config file checks when hot_reload is enabled (default is 60).
    log_level : str, optional
        The logging level to use (default is "INFO").
    log_to_file : bool, optional
        Whether to log output to a file (default is False).
    """
    config_name, config_path = setup_config_file(config_name)
    setup_logging(config_name, log_level, log_to_file)

    try:
        with open(config_path, "r") as f:
            raw_config = json5.load(f)

        if "modes" in raw_config and "default_mode" in raw_config:
            mode_config = load_mode_config(config_name)
            runtime = ModeCortexRuntime(
                mode_config,
                config_name,
                hot_reload=hot_reload,
                check_interval=check_interval,
            )
            logging.info(f"Starting OM1 with mode-aware configuration: {config_name}")
            logging.info(f"Available modes: {list(mode_config.modes.keys())}")
            logging.info(f"Default mode: {mode_config.default_mode}")
        else:
            config = load_config(config_name)
            runtime = CortexRuntime(
                config,
                config_name,
                hot_reload=hot_reload,
                check_interval=check_interval,
            )
            logging.info(f"Starting OM1 with standard configuration: {config_name}")

        if hot_reload:
            logging.info(
                f"Hot-reload enabled (check interval: {check_interval} seconds)"
            )

        asyncio.run(runtime.run())

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)


if __name__ == "__main__":

    # Fix for Linux multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="OM1 Runtime: Launch and manage AI agents with multimodal capabilities. Load agent configurations from the config/ directory to control behavior and capabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Agent Selection
    agent_group = parser.add_argument_group("Agent Selection")
    agent_group.add_argument(
        "config_name",
        nargs="?",
        help="Agent configuration name (e.g., 'spot', 'turtlebot4'). Loads <name>.json5 from config/ directory. If not provided, uses memory/.runtime.json5 as default.",
    )

    # Runtime Options
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument(
        "--hot-reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically reload configuration when files change. Useful for iterative development and tuning.",
    )
    runtime_group.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Interval (seconds) to check for configuration changes. Used only when --hot-reload is enabled.",
    )

    # Debugging & Logging
    debug_group = parser.add_argument_group("Debugging & Logging")
    debug_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level for console output. DEBUG provides detailed trace information.",
    )
    debug_group.add_argument(
        "--log-to-file",
        action="store_true",
        help="Write logs to a file in the logs/ directory in addition to console output.",
    )

    args = parser.parse_args()

    start(
        config_name=args.config_name,
        hot_reload=args.hot_reload,
        check_interval=args.check_interval,
        log_level=args.log_level,
        log_to_file=args.log_to_file,
    )