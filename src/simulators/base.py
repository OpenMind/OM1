import functools
import threading
import time
import typing as T
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from llm.output_model import Action
from providers.prometheus_monitor import PrometheusMonitor


class SimulatorConfig(BaseModel):
    """
    Configuration class for Simulator implementations.
    """

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = Field(default=None, description="Name of the simulator")
    host: Optional[str] = Field(default=None, description="Host address for simulator")
    port: Optional[int] = Field(default=None, description="Port number for simulator")


class Simulator:
    """
    Base class for simulation components.
    """

    def __init__(self, config: SimulatorConfig):
        """
        Initialize simulator with configuration.

        Parameters
        ----------
        config : SimulatorConfig
            Configuration object for the simulator
        """
        self.config = config
        self.name = config.name or "Simulator"
        self._stop_event: Optional[threading.Event] = None
        # Set up Prometheus monitor for subclasses
        self._monitor = PrometheusMonitor()
        self._monitor.register(self.__class__.__name__, metadata={"type": "simulator"})

    def __init_subclass__(cls, **kwargs: T.Any) -> None:
        """Auto-wrap sim with heartbeat reporting."""
        super().__init_subclass__(**kwargs)
        if "sim" in cls.__dict__:
            original = cls.__dict__["sim"]

            @functools.wraps(original)
            def wrapped(self: "Simulator", *args: T.Any, **kw: T.Any) -> T.Any:  # type: ignore
                result = original(self, *args, **kw)
                self._monitor.heartbeat(self.__class__.__name__)
                return result

            cls.sim = wrapped  # type: ignore

    def set_stop_event(self, stop_event: threading.Event) -> None:
        """
        Set the stop event for this simulator.

        Parameters
        ----------
        stop_event : threading.Event
            Event that signals when the simulator should stop
        """
        self._stop_event = stop_event

    def should_stop(self) -> bool:
        """
        Check if the simulator should stop.

        Returns
        -------
        bool
            True if the simulator should stop, False otherwise
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

    def sim(self, actions: List[Action]) -> None:
        """
        Simulate the environment with the given actions.
        """
        pass

    def tick(self) -> None:
        """
        Run the simulator for one tick.

        Note: This method should not block the event loop.
        """
        self.sleep(60)
        pass
