import typing as T
from dataclasses import dataclass

R = T.TypeVar("R")


@dataclass
class SensorConfig:
    """
    Configuration class for Sensor implementations.

    Parameters
    ----------
    **kwargs : dict
        Additional configuration parameters
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class Sensor(T.Generic[R]):
    """
    Base class for all sensors. Provides the interface for converting raw inputs
    into text format for processing by the fuser.

    Type Parameters
    --------------
    R
        The raw input type that this agent handles
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize an Sensor instance.
        """
        self.config = config
        pass

    async def _raw_to_text(self, raw_input: R) -> str:
        """
        Convert raw input data into text format for processing.

        Parameters
        ----------
        raw_input : R
            The raw input data to convert

        Returns
        -------
        str
            Text representation of the input

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError

    async def raw_to_text(self, raw_input: R):
        """
        Convert raw input data into text format for processing.

        Parameters
        ----------
        raw_input : R
            The raw input data to convert

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError

    def formatted_latest_buffer(self) -> str | None:
        """
        Get the most recent input buffer as a formatted prompt string.

        Returns
        -------
        str or None
            The formatted buffer string if available, None otherwise

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError

    async def listen(self) -> T.AsyncIterator[R]:
        """
        Create an asynchronous iterator that yields raw input events.

        The iterator continues until the input stream is closed or an error occurs.

        Yields
        ------
        R
            Raw input events from the source

        Notes
        -----
        This method relies on the _listen_loop() implementation which must be
        provided by subclasses.
        """
        async for event in self._listen_loop():  # type: ignore
            yield event
