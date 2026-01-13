import asyncio
import time
from queue import Empty, Queue
from typing import List, Optional

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.simple_paths_provider import SimplePathsProvider


class SimplePaths(FuserInput[SensorConfig, Optional[str]]):
    """
    Simple paths input handler for robot navigation and obstacle avoidance.

    This class processes simple path data from the SimplePathsProvider and generates
    text descriptions about objects and walls in the robot's environment. It maintains
    an internal buffer of processed messages and integrates with the IO provider for
    logging and tracking purposes.

    The SimplePaths input is designed to help the robot plan movements and avoid
    obstacles by providing structured information about the surrounding environment.
    It continuously polls the SimplePathsProvider for updated path data and converts
    raw input into timestamped messages for downstream processing by the agent's
    input pipeline.

    Typical use cases include:
    - Real-time obstacle detection and avoidance
    - Path planning based on environmental awareness
    - Integration with LLM-based navigation systems
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize the SimplePaths input handler.

        Parameters
        ----------
        config : SensorConfig
            Configuration object containing sensor settings and parameters.
            The config is passed to the parent FuserInput class for initialization.

        Notes
        -----
        The initialization process automatically:
        - Creates an IOProvider instance for input tracking
        - Initializes message buffers for storing processed data
        - Starts the SimplePathsProvider in a background thread
        - Sets up the descriptor string for LLM integration
        """
        super().__init__(config)

        # Track IO
        self.io_provider = IOProvider()

        # Buffer for storing the final output
        self.messages: List[Message] = []

        # Buffer for storing messages
        self.message_buffer: Queue[str] = Queue()

        # Initialize SimplePaths Provider
        self.paths_provider: SimplePathsProvider = SimplePathsProvider()
        self.paths_provider.start()

        self.descriptor_for_LLM = "Information about objects and walls around you, to plan your movements and avoid bumping into things."

    async def _poll(self) -> Optional[str]:
        """
        Poll for new path data from the SimplePaths Provider.

        Periodically checks the SimplePathsProvider for updated path information
        with a brief delay to prevent excessive CPU usage. The method retrieves
        the current lidar string representation of the environment.

        Returns
        -------
        Optional[str]
            The current path data string from the SimplePathsProvider if available,
            None if no data is available or an error occurs
        """
        await asyncio.sleep(0.2)

        try:
            return self.paths_provider.lidar_string
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Creates a Message object from the raw input string, adding
        the current timestamp.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Processes the raw input if present and adds the resulting
        message to the internal message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available
        """
        if raw_input is None:
            return

        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Retrieves the most recent message from the buffer, formats it
        with timestamp and class name, adds it to the IO provider,
        and clears the buffer.

        Returns
        -------
        Optional[str]
            Formatted string containing the latest message and metadata,
            or None if the buffer is empty

        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.descriptor_for_LLM, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result
