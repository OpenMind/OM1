import asyncio

from inputs.base import Sensor


class InputOrchestrator:
    """
    Manages and coordinates multiple input sources.

    Handles concurrent processing of multiple Sensor instances,
    orchestrating their data flows.

    Parameters
    ----------
    inputs : list[Sensor]
        List of input sources to manage
    """

    inputs: list[Sensor]

    def __init__(self, inputs: list[Sensor]):
        """
        Initialize InputOrchestrator instance with input sources.
        """
        self.inputs = inputs

    async def listen(self) -> None:
        """
        Start listening to all input sources concurrently.

        Creates and manages async tasks for each input source.
        """
        input_tasks = []

        for input_source in self.inputs:
            if input_source is None:
                # optional input missing, skip silently
                continue

            try:
                task = asyncio.create_task(self._listen_to_input(input_source))
                input_tasks.append(task)
            except Exception as e:
                print(f"[warning] Optional input failed to start: {type(input_source).__name__}: {e}")
                continue

        if input_tasks:
            await asyncio.gather(*input_tasks)

    async def _listen_to_input(self, input: Sensor) -> None:
        """
        Process events from a single input source.

        Parameters
        ----------
        input : Sensor
            Input source to listen to
        """
        try:
            async for event in input.listen():
                try:
                    await input.raw_to_text(event)
                except Exception as e:
                    print(f"[warning] Failed to process input event: {e}")
                    continue
        except Exception as e:
            print(f"[warning] Input listener stopped due to error ({type(input).__name__}): {e}")
