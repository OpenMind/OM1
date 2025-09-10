import asyncio
import multiprocessing as mp

import dotenv
import typer

from runtime.config import load_config
from runtime.cortex import CortexRuntime
from runtime.logging import setup_logging

app = typer.Typer()


@app.command()
def start(config_name: str, log_level: str = "INFO", log_to_file: bool = False) -> None:
    setup_logging(config_name, log_level, log_to_file)

    # Load configuration
    config = load_config(config_name)
    runtime = CortexRuntime(config)

    # Start the runtime
    asyncio.run(runtime.run())


if __name__ == "__main__":

    # Fix for Linux multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    dotenv.load_dotenv()
    app()
