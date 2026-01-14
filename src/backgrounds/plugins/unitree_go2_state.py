import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_state_provider import UnitreeGo2StateProvider


class UnitreeGo2StateConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 State Background.

    Parameters
    ----------
    unitree_ethernet : Optional[str]
        Unitree Go2 Ethernet channel.
    """

    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Go2 Ethernet channel"
    )


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

        unitree_ethernet = self.config.unitree_ethernet
        if not unitree_ethernet:
            logging.error(
                "Unitree Go2 Ethernet channel is not set in the configuration."
            )
            raise ValueError(
                "Unitree Go2 Ethernet channel must be specified in the configuration."
            )

        self.unitree_go2_state_provider = UnitreeGo2StateProvider()
        logging.info("Unitree Go2 State Provider initialized in background")
