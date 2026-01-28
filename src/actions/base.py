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
    An interface for a action.

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
        time.sleep(60)


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
