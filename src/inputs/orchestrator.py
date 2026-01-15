import asyncio
import logging
from collections.abc import Sequence

from inputs.base import Sensor


class InputOrchestrator:
    """
    Manages and coordinates multiple input sources.

    Handles concurrent processing of multiple Sensor instances,
    orchestrating their data flows. If one input fails, other inputs
    continue to operate independently.

    Parameters
    ----------
    inputs : Sequence[Sensor]
        Sequence of input sources to manage
    """

    inputs: Sequence[Sensor]

    def __init__(self, inputs: Sequence[Sensor]):
        """
        Initialize InputOrchestrator instance with input sources.
        """
        self.inputs = inputs

    async def listen(self) -> None:
        """
        Start listening to all input sources concurrently.

        Creates and manages async tasks for each input source.
        Each input runs independently - if one fails, others continue.
        """
        input_tasks = [
            asyncio.create_task(
                self._listen_to_input_with_error_handling(input), name=f"input-{i}"
            )
            for i, input in enumerate(self.inputs)
        ]
        await asyncio.gather(*input_tasks, return_exceptions=True)

    async def _listen_to_input_with_error_handling(self, input: Sensor) -> None:
        """
        Wrapper for _listen_to_input that handles errors gracefully.

        If an input fails, it logs the error but doesn't stop other inputs.

        Parameters
        ----------
        input : Sensor
            Input source to listen to
        """
        input_name = type(input).__name__
        try:
            await self._listen_to_input(input)
        except asyncio.CancelledError:
            logging.info(f"Input '{input_name}' was cancelled")
            raise
        except Exception as e:
            logging.error(
                f"Input '{input_name}' failed with error: {type(e).__name__}: {e}",
                exc_info=True,
            )
            # Don't re-raise - let other inputs continue

    async def _listen_to_input(self, input: Sensor) -> None:
        """
        Process events from a single input source.

        Parameters
        ----------
        input : Sensor
            Input source to listen to
        """
        async for event in input.listen():
            await input.raw_to_text(event)
