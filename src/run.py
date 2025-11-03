import asyncio
import logging
import multiprocessing as mp
import os

import dotenv
import json5
import typer

from runtime.logging import setup_logging
from runtime.multi_mode.config import load_mode_config
from runtime.multi_mode.cortex import ModeCortexRuntime
from runtime.single_mode.config import load_config
from runtime.single_mode.cortex import CortexRuntime

app = typer.Typer()


@app.command()
def start(
    config_name: str,
    hot_reload: bool = True,
    check_interval: int = 60,
    log_level: str = "INFO",
    log_to_file: bool = False,
) -> None:
    """
    Start the OM1 agent with a specific configuration.

    Parameters
    ----------
    config_name : str
        The name of the configuration file (without extension) located in the config directory.
    hot_reload : bool, optional
        Enable hot-reload of configuration files (default is True).
    check_interval : int, optional
        Interval in seconds between config file checks when hot_reload is enabled (default is 60).
    log_level : str, optional
        The logging level to use (default is "INFO").
    log_to_file : bool, optional
        Whether to log output to a file (default is False).
    """
    setup_logging(config_name, log_level, log_to_file)

    config_path = os.path.join(
        os.path.dirname(__file__), "../config", config_name + ".json5"
    )

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
        raise typer.Exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":

    # Fix for Linux multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    dotenv.load_dotenv()
    app()
