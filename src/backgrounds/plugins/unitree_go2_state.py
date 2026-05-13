import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_state_provider import UnitreeGo2StateProvider
from providers.unitree_go2_state_zenoh_provider import UnitreeGo2StateZenohProvider


class UnitreeGo2StateConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 State Background.

    Parameters
    ----------
    api_key : Optional[str]
        API Key for OpenMind cloud system, if required.
    use_sim : bool
        Whether to use the simulation Zenoh endpoint instead of a local one.
    unitree_ethernet : Optional[str]
        Unitree Go2 Ethernet channel.
    """

    api_key: Optional[str] = Field(default=None, description="API Key for OpenMind cloud system, if required.")
    use_sim: bool = Field(
        default=False,
        description="Whether to use the simulation Zenoh endpoint instead of a local one.",
    )
    unitree_ethernet: Optional[str] = Field(default=None, description="Unitree Go2 Ethernet channel")


class UnitreeGo2State(Background[UnitreeGo2StateConfig]):
    """
    Background task for reading and monitoring Unitree Go2 robot state data.

    This background task initializes and manages a UnitreeGo2StateProvider
    that continuously monitors the robot's internal state through the Unitree
    Ethernet communication channel. The provider tracks various robot state
    parameters including joint positions, velocities, battery status, and
    operational modes.

    The state data is essential for real-time robot control, safety monitoring,
    and adaptive behavior planning in Unitree Go2 robot applications. The
    provider ensures continuous state updates for responsive robot interactions.
    """

    def __init__(self, config: UnitreeGo2StateConfig):
        """
        Initialize the Unitree Go2 State background task.

        Parameters
        ----------
        config : UnitreeGo2StateConfig
            Configuration for the Unitree Go2 State background task.
        """
        super().__init__(config)

        if self.config.use_sim:
            self.unitree_go2_state_provider = UnitreeGo2StateZenohProvider(self.config.api_key, self.config.use_sim)
            logging.info("Unitree Go2 State Zenoh Provider initialized in background")
            return

        unitree_ethernet = self.config.unitree_ethernet
        if not unitree_ethernet:
            logging.error("Unitree Go2 Ethernet channel is not set in the configuration.")
            raise ValueError("Unitree Go2 Ethernet channel must be specified in the configuration.")

        self.unitree_go2_state_provider = UnitreeGo2StateProvider()
        logging.info("Unitree Go2 State Provider initialized in background")
