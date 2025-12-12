import time
import typing as T
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

IT = T.TypeVar("IT")
OT = T.TypeVar("OT")


@dataclass
class MoveCommand:
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
    """

    input: IT
    output: OT


class ActionConnector(ABC, T.Generic[OT]):
    def __init__(self, config: ActionConfig):
        self.config = config

    @abstractmethod
    async def connect(self, output_interface: OT) -> None:
        pass

    def tick(self) -> None:
        time.sleep(60)


@dataclass
class AgentAction:
    """Base class for agent actions"""

    name: str
    llm_label: str
    interface: T.Type[Interface]
    connector: ActionConnector
    exclude_from_prompt: bool
