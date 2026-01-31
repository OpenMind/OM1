import threading
import time
import typing as T

from pydantic import BaseModel, ConfigDict

ConfigType = T.TypeVar("ConfigType", bound="BackgroundConfig")


class BackgroundConfig(BaseModel):
    """
    Base configuration class for Background implementations.
    """

    model_config = ConfigDict(extra="allow")


class Background(T.Generic[ConfigType]):
    """
    Base class for background components.
    """

    def __init__(self, config: ConfigType):
        """
        Initialize background with configuration.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background
        """
        self.config = config

        self.name = getattr(config, "name", type(self).__name__)

        self._stop_event: T.Optional[threading.Event] = None

    def set_stop_event(self, stop_event: threading.Event) -> None:
        """
        Set the stop event for this background task.

        Parameters
        ----------
        stop_event : threading.Event
            Event that signals when the background task should stop
        """
        self._stop_event = stop_event

    def should_stop(self) -> bool:
        """
        Check if the background task should stop.

        Returns
        -------
        bool
            True if the task should stop, False otherwise
        """
        return self._stop_event is not None and self._stop_event.is_set()

    def sleep(self, duration: float) -> bool:
        """
        Sleep for the specified duration, but wake immediately if stop signal is received.

        Parameters
        ----------
        duration : float
            Total duration to sleep in seconds

        Returns
        -------
        bool
            True if sleep completed normally, False if interrupted by stop signal
        """
        if self._stop_event is None:
            time.sleep(duration)
            return True

        was_stopped = self._stop_event.wait(timeout=duration)

        return not was_stopped

    def run(self) -> None:
        """
        Run the background process.

        This method should be overridden by subclasses to implement specific behavior.
        """
        self.sleep(60)
