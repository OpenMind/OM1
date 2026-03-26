import asyncio
import logging
import multiprocessing as mp
import os
import shutil
from typing import Optional, Tuple

import dotenv
import typer

from runtime.config import load_mode_config
from runtime.cortex import ModeCortexRuntime
from runtime.logging import setup_logging

app = typer.Typer()


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
            raise typer.Exit(1)

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


@app.command()
def start(
    config_name: Optional[str] = typer.Argument(
        None,
        help="The name of the configuration file (without extension) located in the config directory. If not provided, uses .runtime.json5 from memory folder.",
    ),
    hot_reload: bool = typer.Option(
        True, help="Enable hot-reload of configuration files."
    ),
    check_interval: int = typer.Option(
        60,
        help="Interval in seconds between config file checks when hot_reload is enabled.",
    ),
    log_level: str = typer.Option("INFO", help="The logging level to use."),
    log_to_file: bool = typer.Option(False, help="Whether to log output to a file."),
    collect_data: bool = typer.Option(False, envvar="OM1_COLLECT_DATA", help="Enable background data collection (Video, Audio, Lidar)."),
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
    collect_data : bool, optional
        Enable background data collection in an isolated process (default is False).
    """
    config_name, config_path = setup_config_file(config_name)
    setup_logging(config_name, log_level, log_to_file)

    try:
        mode_config = load_mode_config(config_name)
        runtime = ModeCortexRuntime(
            mode_config,
            config_name,
            hot_reload=hot_reload,
            check_interval=check_interval,
        )
        logging.info(f"Starting OM1 with configuration: {config_name}")
        logging.info(f"Available modes: {list(mode_config.modes.keys())}")
        logging.info(f"Default mode: {mode_config.default_mode}")

        if hot_reload:
            logging.info(
                f"Hot-reload enabled (check interval: {check_interval} seconds)"
            )

        if collect_data:
            import atexit
            import json5
            from data_collector import run_data_collector_process

            dc_config: dict = {}
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        raw = json5.load(f)
                        dc_config = raw.get("data_collector", {})
                except Exception as e:
                    logging.warning(f"Could not parse data_collector config from json5: {e}")

            v_rtsp = dc_config.get("video_rtsp", "rtsp://localhost:8554/top_camera")
            a_rtsp = dc_config.get("audio_rtsp", "rtsp://localhost:8554/audio")
            lidar_p = dc_config.get("lidar_port", "/dev/ttyUSB0")
            odom_c = dc_config.get("odom_channel") or mode_config.unitree_ethernet
            rollover_c = dc_config.get("rollover_seconds", 120)

            logging.info("Starting isolated data collector process...")
            logging.info(f"Collector Config -> Video: {v_rtsp}, Audio: {a_rtsp}, LiDAR: {lidar_p}, Odom: {odom_c}, Rollover: {rollover_c}s")
            
            collector_p = mp.Process(
                target=run_data_collector_process,
                args=(v_rtsp, a_rtsp, lidar_p, odom_c, rollover_c),
                daemon=False,  # False so it's not abruptly killed by OS
            )
            collector_p.start()

            def cleanup_collector():
                if collector_p.is_alive():
                    logging.info("Sending graceful termination signal to data collector...")
                    collector_p.terminate()  # Sends SIGTERM
                    collector_p.join(timeout=6.0)

            atexit.register(cleanup_collector)

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
