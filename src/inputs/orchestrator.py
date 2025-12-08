import asyncio

from inputs.base import Sensor


class InputOrchestrator:
    """
    Manages and coordinates multiple input sources.

    Handles concurrent processing of multiple Sensor instances,
    orchestrating their data flows and managing async operations
    for real-time input processing.

    Parameters
    ----------
    inputs : list[Sensor]
        List of input sources to manage and coordinate

    Attributes
    ----------
    inputs : list[Sensor]
        Collection of sensor input sources being managed
    """

    inputs: list[Sensor]

    def __init__(self, inputs: list[Sensor]):
        """
        Initialize InputOrchestrator instance with input sources.

        Sets up the orchestrator with the provided list of input sensors
        and prepares for concurrent processing operations.

        Parameters
        ----------
        inputs : list[Sensor]
            List of sensor input sources to be managed by this orchestrator
        """
        self.inputs = inputs

    async def listen(self) -> None:
        """
        Start listening to all input sources concurrently.

        Creates and manages async tasks for each input source, allowing
        parallel processing of multiple sensor inputs. This method blocks
        until all input sources complete or encounter an error.

        The method uses asyncio.gather to run all input listeners concurrently,
        providing better performance for multiple input sources.

        Raises
        ------
        Exception
            If any of the input tasks encounter an error during execution
        """
        input_tasks = [
            asyncio.create_task(self._listen_to_input(input)) for input in self.inputs
        ]
        await asyncio.gather(*input_tasks)

    async def _listen_to_input(self, input: Sensor) -> None:
        """
        Process events from a single input source.

        Continuously listens to a single sensor input source and processes
        incoming events by converting them to text format. This method runs
        indefinitely until the input source is closed or an error occurs.

        Parameters
        ----------
        input : Sensor
            Input source to listen to and process events from

        Notes
        -----
        This method implements an async iterator pattern to handle
        streaming input data from sensors in real-time.
        """
        async for event in input.listen():
            await input.raw_to_text(event)
