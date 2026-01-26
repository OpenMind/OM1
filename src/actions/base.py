import time
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

IT = T.TypeVar("IT")
OT = T.TypeVar("OT")


@dataclass
class MoveCommand:
    """Represents a movement command for robot locomotion.

    Attributes
    ----------
    dx : float
        Forward/backward displacement in meters.
    yaw : float
        Rotation angle in radians.
    start_x : float
        Starting X position, defaults to 0.0.
    start_y : float
        Starting Y position, defaults to 0.0.
    turn_complete : bool
        Whether the turn has been completed, defaults to False.
    speed : float
        Movement speed multiplier, defaults to 0.5.
    """

    dx: float
    yaw: float
    start_x: float = 0.0
    start_y: float = 0.0
    turn_complete: bool = False
    speed: float = 0.5


@dataclass
class ActionConfig:
    """Configuration class for Action implementations.

    Parameters
    ----------
    **kwargs : dict
        Additional configuration parameters
    """

    def __init__(self, **kwargs):
        """Initialize ActionConfig with dynamic attributes.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments that will be set as instance attributes.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class Interface(T.Generic[IT, OT]):
    """Generic interface for actions.

    Attributes
    ----------
    input : IT
        The input type for the action.
    output : OT
        The output type for the action.
    """

    input: IT
    output: OT


class ActionConnector(ABC, T.Generic[OT]):
    """Abstract base class for action connectors.

    Action connectors handle the communication between the agent's
    action decisions and the actual hardware or simulation endpoints.
    """

    def __init__(self, config: ActionConfig):
        """Initialize the action connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration object containing connector-specific settings.
        """
        self.config = config

    @abstractmethod
    async def connect(self, input_protocol: OT) -> None:
        """Send an action command to the connected endpoint.

        This method must be implemented by subclasses to handle
        the actual transmission of action commands.

        Parameters
        ----------
        input_protocol : OT
            The action input data to be sent to the endpoint.
        """
        pass

    def tick(self) -> None:
        """Execute a single iteration of the connector's main loop.

        This default implementation sleeps for 60 seconds. Subclasses
        should override this method to implement custom tick behavior
        with appropriate timing for their specific use case.
        """
        time.sleep(60)


@dataclass
class AgentAction:
    """Base class for agent actions.

    Attributes
    ----------
    name : str
        The name identifier for this action.
    llm_label : str
        The label used by the LLM to reference this action.
    interface : Type[Interface]
        The interface type for this action.
    connector : ActionConnector
        The connector instance handling action execution.
    exclude_from_prompt : bool
        Whether to exclude this action from LLM prompts.
    """

    name: str
    llm_label: str
    interface: T.Type[Interface]
    connector: ActionConnector
    exclude_from_prompt: bool
