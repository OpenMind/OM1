import asyncio
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
        """
        self.inputs = inputs

    async def listen(self) -> None:
        """
        Start listening to all input sources concurrently.

        Creates and manages async tasks for each input source.
        """
        input_tasks = [
            asyncio.create_task(self._listen_to_input(input)) for input in self.inputs
        ]
        await asyncio.gather(*input_tasks)

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
