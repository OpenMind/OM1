import asyncio
import logging
from collections.abc import Sequence

from inputs.base import Sensor
from runtime.config import RuntimeConfig


class InputOrchestrator:
    """
    Manages and coordinates multiple input sources.
    """

    inputs: Sequence[Sensor]

    def __init__(self, inputs: Sequence[Sensor]):
        self.inputs = inputs

    async def listen(self) -> None:
        """Start listening to all input sources concurrently."""
        input_tasks = [
            asyncio.create_task(
                self._listen_to_input(input), name=f"input_{type(input).__name__}"
            )
            for input in self.inputs
        ]
        results = await asyncio.gather(*input_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                input_name = type(self.inputs[i]).__name__
                logging.error(f"Input {input_name} failed with error: {result}")

    async def _listen_to_input(self, input: Sensor) -> None:
        input_name = type(input).__name__
        try:
            async for event in input.listen():
                try:
                    await input.raw_to_text(event)
                except Exception as e:
                    logging.error(
                        f"Error processing event in {input_name}: {e}", exc_info=True
                    )
        except Exception as e:
            logging.error(f"Input {input_name} listener failed: {e}", exc_info=True)
            raise

    def update_config(self, new_config: RuntimeConfig):
        """
        Update configuration for inputs.
        Currently does not support dynamic reconfiguration.
        """
        logging.warning("InputOrchestrator does not support dynamic config update.")
