from pathlib import Path
import os
import asyncio
import logging
import multiprocessing as mp
import shutil
from typing import Optional, Tuple

import dotenv
import json5
import typer

from config_manager import (
    ConfigHotReloader,
    ConfigWatcher,
    resolve_hot_reload_spec,
)

from runtime.logging import setup_logging
from runtime.single_mode.config import load_config
from runtime.single_mode.cortex import CortexRuntime

# Multi-mode is OPTIONAL: avoid import-time crash if zenoh is missing
try:
    from runtime.multi_mode.config import load_mode_config
    from runtime.multi_mode.cortex import ModeCortexRuntime

    MULTI_MODE_AVAILABLE = True
except Exception:
    MULTI_MODE_AVAILABLE = False

app = typer.Typer()


def setup_config_file(config_name: Optional[str]) -> Tuple[str, str]:
    if config_name is None:
        runtime_config_path = os.path.join(
            os.path.dirname(__file__), "../config/memory", ".runtime.json5"
        )

        if not os.path.exists(runtime_config_path):
            logging.error(
                f"Default runtime configuration file not found: {runtime_config_path}"
            )
            raise typer.Exit(1)

        config_name = ".runtime"
        config_path = os.path.join(
            os.path.dirname(__file__), "../config", config_name + ".json5"
        )

        shutil.copy2(runtime_config_path, config_path)
        logging.info("Using default runtime configuration from memory folder")
    else:
        config_path = os.path.join(
            os.path.dirname(__file__), "../config", config_name + ".json5"
        )

    return config_name, config_path


@app.command()
def start(
    config_name: Optional[str] = typer.Argument(
        None,
        help="Config name (without .json5) from config/ directory.",
    ),
    hot_reload: bool = typer.Option(
        True,
        help="Enable hot-reload (polling + selective watchdog reload).",
    ),
    check_interval: int = typer.Option(
        60,
        help="Polling interval in seconds for legacy hot-reload.",
    ),
    log_level: str = typer.Option("INFO", help="Logging level."),
    log_to_file: bool = typer.Option(False, help="Log to file."),
) -> None:
    config_name, config_path = setup_config_file(config_name)
    setup_logging(config_name, log_level, log_to_file)

    watcher: Optional[ConfigWatcher] = None

    try:
        # Load raw JSON5 (needed for hot-reload spec)
        with open(config_path, "r") as f:
            raw_config = json5.load(f)

        # Create runtime
        if (
            MULTI_MODE_AVAILABLE
            and "modes" in raw_config
            and "default_mode" in raw_config
        ):
            mode_config = load_mode_config(config_name)
            runtime = ModeCortexRuntime(
                mode_config,
                config_name,
                hot_reload=hot_reload,
                check_interval=check_interval,
            )
            logging.info(f"Starting OM1 (multi-mode): {config_name}")
        else:
            config = load_config(config_name)
            runtime = CortexRuntime(
                config,
                config_name,
                hot_reload=hot_reload,
                check_interval=check_interval,
            )
            logging.info(f"Starting OM1 (single-mode): {config_name}")

        # ------------------------------------------------------------
        # 🔥 Selective hot-reload via watchdog (NEW FEATURE)
        # ------------------------------------------------------------
        try:
            spec = resolve_hot_reload_spec(raw_config)

            if hot_reload:
                spec = spec.__class__(enabled=True, fields=spec.fields)

            if spec.enabled and spec.fields:
                debounce_ms = int(os.getenv("HOT_RELOAD_DEBOUNCE_MS", "300"))

                reloader = ConfigHotReloader(
                    config_path=Path(config_path),
                    runtime=runtime,
                    spec=spec,
                )
                reloader.initialize()

                watcher = ConfigWatcher(
                    config_path=Path(config_path),
                    reloader=reloader,
                    debounce_ms=debounce_ms,
                )
                watcher.start()

                logging.info(
                    f"Selective hot-reload enabled for fields: {list(spec.fields)}"
                )

        except Exception:
            logging.exception(
                "Failed to initialize selective hot-reload (continuing without it)"
            )

        asyncio.run(runtime.run())

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise typer.Exit(1)
    except Exception as e:
        logging.exception(f"Error starting OM1: {e}")
        raise typer.Exit(1)
    finally:
        if watcher:
            watcher.stop()


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    dotenv.load_dotenv()
    app()
