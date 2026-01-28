import asyncio
import logging
from collections.abc import Sequence

from inputs.base import Sensor


class InputOrchestrator:
    """
    Manages and coordinates multiple input sources.

    Handles concurrent processing of multiple Sensor instances,
    orchestrating their data flows.

    Parameters
    ----------
    inputs : Sequence[Sensor]
        Sequence of input sources to manage
    """

    inputs: Sequence[Sensor]

    def __init__(self, inputs: Sequence[Sensor]):
        """
        Initialize InputOrchestrator instance with input sources.

        Parameters
        ----------
        inputs : Sequence[Sensor]
            Sequence of input sources to manage.
        """
        self.inputs = inputs

    async def listen(self) -> None:
        """
        Start listening to all input sources concurrently.

        Creates and manages async tasks for each input source.
        If one input fails, other inputs continue operating.
        """
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
        """
        Process events from a single input source.

        Parameters
        ----------
        input : Sensor
            Input source to listen to
        """
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
