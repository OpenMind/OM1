import time
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from llm.output_model import Action


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
        time.sleep(60)
        pass
