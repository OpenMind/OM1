import asyncio
from collections.abc import Sequence

from inputs.base import Sensor


class InputOrchestrator:
    """
    Manages and coordinates multiple input sources.

    Handles concurrent processing of multiple Sensor instances,
    orchestrating their data flows. Each input source is processed
    asynchronously in parallel, allowing for efficient handling of
    multiple sensors simultaneously.

    The orchestrator creates separate async tasks for each input source
    and coordinates their execution using asyncio.gather(). Events from
    each sensor are processed through the raw_to_text() method.

    Parameters
    ----------
    inputs : Sequence[Sensor]
        Sequence of input sources to manage. Each sensor should implement
        the Sensor interface with listen() and raw_to_text() methods.
    """

    inputs: Sequence[Sensor]

    def __init__(self, inputs: Sequence[Sensor]):
        """
        Initialize InputOrchestrator instance with input sources.

        Parameters
        ----------
        inputs : Sequence[Sensor]
            Sequence of input sources to manage. Each sensor should implement
            the Sensor interface with listen() and raw_to_text() methods.
        """
        self.inputs = inputs

    async def listen(self) -> None:
        """
        Start listening to all input sources concurrently.

        Creates and manages async tasks for each input source, running
        them in parallel using asyncio.gather(). This method will run
        indefinitely until all input sources stop producing events.

        Notes
        -----
        Each input source is processed in a separate async task, allowing
        for true concurrent processing. Events from each sensor are
        automatically converted to text via the raw_to_text() method.
        """
        input_tasks = [
            asyncio.create_task(self._listen_to_input(input)) for input in self.inputs
        ]
        await asyncio.gather(*input_tasks)

    async def _listen_to_input(self, input: Sensor) -> None:
        """
        Process events from a single input source.

        Continuously listens to events from the specified input source
        and processes them through the raw_to_text() method. This method
        runs until the input source stops producing events.

        Parameters
        ----------
        input : Sensor
            Input source to listen to. Must implement the Sensor interface
            with listen() and raw_to_text() methods.
        """
        async for event in input.listen():
            await input.raw_to_text(event)
