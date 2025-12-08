import time
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

IT = T.TypeVar("IT")
OT = T.TypeVar("OT")


@dataclass
class MoveCommand:
    """
    Command structure for robot movement operations.

    This dataclass represents a movement command with position, orientation,
    and execution parameters for robot navigation.

    Parameters
    ----------
    dx : float
        Delta x coordinate for movement (meters)
    yaw : float
        Yaw angle for rotation (radians)
    start_x : float, default 0.0
        Starting x position for the movement
    start_y : float, default 0.0
        Starting y position for the movement
    turn_complete : bool, default False
        Flag indicating if the turn operation is complete
    speed : float, default 0.5
        Movement speed multiplier (0.0 to 1.0)
    """

    dx: float
    yaw: float
    start_x: float = 0.0
    start_y: float = 0.0
    turn_complete: bool = False
    speed: float = 0.5


@dataclass
class ActionConfig:
    """
    Configuration class for Action implementations.

    Parameters
    ----------
    **kwargs : dict
        Additional configuration parameters
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class Interface(T.Generic[IT, OT]):
    """
    Generic interface definition for action input/output types.

    This class defines the contract between input and output types for
    action implementations, ensuring type safety across the action system.

    Parameters
    ----------
    input : IT
        Input type for the action interface
    output : OT
        Output type for the action interface

    Type Parameters
    ---------------
    IT : TypeVar
        Input type variable for generic typing
    OT : TypeVar
        Output type variable for generic typing
    """

    input: IT
    output: OT


class ActionConnector(ABC, T.Generic[OT]):
    """
    Abstract base class for action connectors.

    Action connectors handle the communication between the action system
    and external hardware/software interfaces. Each connector implements
    specific protocols for different robot platforms or services.

    Parameters
    ----------
    config : ActionConfig
        Configuration object containing connector-specific settings

    Type Parameters
    ---------------
    OT : TypeVar
        Output type for the connector interface
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the action connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration object for the connector
        """
        self.config = config

    @abstractmethod
    async def connect(self, input_protocol: OT) -> None:
        """
        Connect to the external interface and execute the action.

        This method must be implemented by subclasses to handle
        the specific communication protocol with the target system.

        Parameters
        ----------
        input_protocol : OT
            Input data to be sent to the external interface

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass
        """
        pass

    def tick(self) -> None:
        """
        Perform periodic maintenance or heartbeat operations.

        Default implementation sleeps for 60 seconds. Subclasses
        should override this method for custom timing requirements.
        """
        time.sleep(60)


@dataclass
class AgentAction:
    """
    Configuration and metadata for a specific agent action.

    This class encapsulates all the information needed to define and execute
    an action within the agent system, including its interface, connector,
    and behavioral configuration.

    Parameters
    ----------
    name : str
        Internal name identifier for the action
    llm_label : str
        Human-readable label for language model interactions
    interface : Type[Interface]
        Interface definition specifying input/output types
    connector : ActionConnector
        Connector instance for handling action execution
    exclude_from_prompt : bool
        Whether to exclude this action from LLM prompts

    Examples
    --------
    >>> config = ActionConfig(host="localhost", port=8080)
    >>> connector = SomeActionConnector(config)
    >>> action = AgentAction(
    ...     name="speak",
    ...     llm_label="speak",
    ...     interface=SpeakInterface,
    ...     connector=connector,
    ...     exclude_from_prompt=False
    ... )
    """

    name: str
    llm_label: str
    interface: T.Type[Interface]
    connector: ActionConnector
    exclude_from_prompt: bool
