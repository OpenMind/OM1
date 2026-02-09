import typing as T

from inputs.base import Sensor, SensorConfig

R = T.TypeVar("R")
ConfigType = T.TypeVar("ConfigType", bound="SensorConfig")


class FuserInput(Sensor[ConfigType, R]):
    """
    Input polling implementation using a continuous asynchronous loop.

    Repeatedly polls for input events in an async loop, yielding results
    as they become available.
    """

    def __init__(self, config: ConfigType):
        """
        Initialize FuserInput instance.

        Parameters
        ----------
        config : ConfigType
            Configuration settings for the FuserInput.
        """
        super().__init__(config)

    async def _listen_loop(self) -> T.AsyncIterator[R]:
        """
        Main polling loop that continuously yields input events.

        Yields
        ------
        R
            Raw input events from polling
        """
        while True:
            yield await self._poll()

    async def _poll(self) -> R:
        """
        Poll for next input event.

        Returns
        -------
        R
            Raw input event

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses
        """
        raise NotImplementedError
