import threading
import time
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

IT = T.TypeVar("IT")
OT = T.TypeVar("OT")
CT = T.TypeVar("CT", bound="ActionConfig")


@dataclass
class MoveCommand:
    """
    Move command interface.

    Parameters
    ----------
    dx : float
        Distance to move in the x direction.
    yaw : float
        Yaw angle to turn.
    start_x : float
        Starting x position.
    start_y : float
        Starting y position.
    turn_complete : bool
        Whether the turn is complete.
    speed : float
        Speed of movement.
    """

    dx: float
    yaw: float
    start_x: float = 0.0
    start_y: float = 0.0
    turn_complete: bool = False
    speed: float = 0.5


class ActionConfig(BaseModel):
    """
    Configuration class for Action implementations.
    """

    model_config = ConfigDict(extra="allow")


@dataclass
class Interface(T.Generic[IT, OT]):
    """
    An interface for an action.

    Parameters
    ----------
    input : IT
        The input type for the interface.
    output : OT
        The output type for the interface.
    """

    input: IT
    output: OT


class ActionConnector(ABC, T.Generic[CT, OT]):
    """
    A connector for an action.
    """

    def __init__(self, config: CT):
        """
        Initialize the ActionConnector.

        Parameters
        ----------
        config : CT
            Configuration for the action connector.
        """
        self.config: CT = config
        self._stop_event: T.Optional[threading.Event] = None

    def set_stop_event(self, stop_event: threading.Event) -> None:
        """
        Set the stop event for this action connector.

        Parameters
        ----------
        stop_event : threading.Event
            Event that signals when the connector should stop
        """
        self._stop_event = stop_event

    def should_stop(self) -> bool:
        """
        Check if the connector should stop.

        Returns
        -------
        bool
            True if the connector should stop, False otherwise
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

    @abstractmethod
    async def connect(self, output_interface: OT) -> None:
        """
        Connect the input protocol to the action.

        Parameters
        ----------
        output_interface : OT
            The input protocol containing the action details.
        """
        pass

    def tick(self) -> None:
        """
        Tick method for periodic updates.
        """
        self.sleep(60)


@dataclass
class AgentAction:
    """
    Base class for agent actions.

    Parameters
    ----------
    name : str
        The name of the action.
    llm_label : str
        The label used by the LLM for this action.
    interface : Type[Interface]
        The interface type for this action.
    connector : ActionConnector
        The connector for this action.
    exclude_from_prompt : bool
        Whether to exclude this action from the prompt.
    """

    name: str
    llm_label: str
    interface: T.Type[Interface]
    connector: ActionConnector
    exclude_from_prompt: bool
