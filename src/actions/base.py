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
    
    This class serves as the base configuration model for all action implementations
    within the system. It utilizes Pydantic's BaseModel to provide type validation
    and serialization capabilities, while allowing additional fields to be dynamically
    added through the `extra="allow"` configuration directive. This design pattern
    enables flexible configuration management where action-specific parameters can
    be extended without modifying the base class structure.
    
    Attributes
    ----------
    model_config : ConfigDict
        Pydantic configuration dictionary that permits additional fields beyond
        those explicitly defined in the model schema, facilitating extensibility
        for action-specific configuration requirements.
    """

    model_config = ConfigDict(extra="allow")


@dataclass
class Interface(T.Generic[IT, OT]):
    """
    Generic interface definition for action input and output type specifications.
    
    This dataclass provides a type-safe mechanism for defining the input and output
    type constraints for actions within the system. By utilizing Python's generic
    type system, it enables compile-time type checking and ensures that actions
    maintain consistent type contracts throughout their execution lifecycle. The
    interface serves as a contract specification that defines the expected data
    structures for both incoming action requests and outgoing action responses.

    Parameters
    ----------
    input : IT
        The input type parameter that specifies the expected data structure or
        type for incoming action requests. This generic type parameter allows
        for flexible specification of input contracts across different action
        implementations.
    output : OT
        The output type parameter that defines the structure and type of data
        that will be produced as a result of action execution. This generic
        parameter enables type-safe handling of action responses and return values.
    """

    input: IT
    output: OT


class ActionConnector(ABC, T.Generic[CT, OT]):
    """
    Abstract base class for action connectors that facilitate communication between
    actions and their underlying execution mechanisms.
    
    This class provides a standardized interface for implementing connectors that
    bridge the gap between the action abstraction layer and the concrete execution
    backends (such as ROS2, Unitree SDK, or other robotic frameworks). The connector
    pattern enables decoupling of action logic from implementation details, allowing
    for flexible substitution of execution mechanisms without modifying action
    definitions. The generic type parameters ensure type safety throughout the
    connector implementation lifecycle.

    Parameters
    ----------
    config : CT
        Configuration object that contains all necessary parameters and settings
        required for the connector to establish and maintain connections with
        the underlying execution backend. The configuration type is constrained
        to be a subclass of ActionConfig, ensuring consistency across connector
        implementations.
    """

    def __init__(self, config: CT):
        """
        Initialize the ActionConnector with the provided configuration object.
        
        This constructor establishes the foundational state of the connector by
        storing the configuration parameters that will govern its behavior during
        connection establishment and action execution. The configuration is
        stored as an instance variable to enable runtime access to connector
        settings throughout the connector's lifecycle.

        Parameters
        ----------
        config : CT
            Configuration object that specifies the operational parameters for
            this connector instance. The configuration must conform to the
            ActionConfig base class structure and may contain connector-specific
            extensions as permitted by the Pydantic model configuration.
        """
        self.config: CT = config

    @abstractmethod
    async def connect(self, output_interface: OT) -> None:
        """
        Establish a connection between the action and its execution backend using
        the provided output interface specification.
        
        This abstract method must be implemented by concrete connector subclasses
        to define the specific mechanism for establishing communication channels
        with the underlying execution system. The method receives an output interface
        that defines the expected data structure and type constraints for action
        execution results, which the connector must utilize to properly format
        and transmit action commands to the backend system.

        Parameters
        ----------
        output_interface : OT
            The output interface specification that defines the type and structure
            of data that will be produced as a result of action execution. This
            interface serves as a contract that the connector must adhere to when
            formatting and transmitting action commands to the execution backend.
            The generic type parameter OT ensures type safety throughout the
            connection establishment process.
        """
        pass

    def tick(self) -> None:
        """
        Execute periodic maintenance and update operations for the connector.
        
        This method provides a mechanism for connectors to perform periodic
        housekeeping tasks, such as connection health checks, resource cleanup,
        or state synchronization with the execution backend. The default
        implementation introduces a 60-second delay, which can be overridden
        by subclasses to implement connector-specific periodic update logic.
        This method is typically invoked by the orchestrator as part of the
        action execution lifecycle management.

        Notes
        -----
        Subclasses should override this method to implement connector-specific
        periodic update logic. The default implementation serves as a placeholder
        that prevents immediate return and may be suitable for connectors that
        do not require periodic maintenance operations.
        """
        time.sleep(60)


@dataclass
class AgentAction:
    """
    Comprehensive action definition that encapsulates all metadata and execution
    components required for an agent to perform a specific action within the system.
    
    This dataclass serves as the primary data structure for representing actions
    that can be executed by autonomous agents. It combines action identification
    metadata (name and LLM label), type specifications (interface), execution
    mechanisms (connector), and behavioral configuration (prompt exclusion) into
    a unified representation. The design enables the action orchestrator to
    manage and execute actions while maintaining clear separation between action
    definitions and their concrete implementations.

    Parameters
    ----------
    name : str
        The internal identifier for this action, used for programmatic reference
        and action lookup within the system. This name should be unique within
        the action registry and typically follows a snake_case naming convention
        for consistency with Python coding standards.
    llm_label : str
        The human-readable label that is presented to the large language model
        (LLM) when describing this action in natural language contexts. This
        label is used in prompt generation to enable the LLM to understand and
        reference the action in conversational interactions, and may differ from
        the internal name to provide more intuitive descriptions.
    interface : Type[Interface]
        The type specification that defines the input and output type constraints
        for this action. This parameter accepts a class type (not an instance)
        that conforms to the Interface generic class structure, establishing
        compile-time type safety and runtime validation for action inputs and
        outputs throughout the execution pipeline.
    connector : ActionConnector
        The concrete connector instance that provides the implementation mechanism
        for executing this action. The connector handles the translation between
        the action abstraction and the underlying execution backend, enabling
        decoupled action definitions that can be executed across different
        robotic frameworks and hardware platforms.
    exclude_from_prompt : bool
        A boolean flag that determines whether this action should be included in
        the prompt that is presented to the LLM. When set to True, the action
        will not be listed as an available option in LLM-generated action
        sequences, effectively making it a system-level action that cannot be
        directly invoked through conversational interactions. This mechanism
        enables the implementation of internal actions that support system
        functionality without exposing them to end users.
    """

    name: str
    llm_label: str
    interface: T.Type[Interface]
    connector: ActionConnector
    exclude_from_prompt: bool
