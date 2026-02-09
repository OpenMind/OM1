import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_odom_provider import UnitreeGo2OdomProvider


class UnitreeGo2OdomConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 Odom Background.

    Parameters
    ----------
    unitree_ethernet : Optional[str]
        Unitree Ethernet channel.
    """

    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Ethernet channel"
    )


class UnitreeGo2Odom(Background[UnitreeGo2OdomConfig]):
    """
    Background task for reading odometry data from Unitree Go2 provider.

    This background task initializes and manages an UnitreeGo2OdomProvider instance
    that reads robot odometry data. The provider can connect via Zenoh
    for distributed robot systems or via Unitree Ethernet for direct
    hardware communication.

    Odometry data includes position, orientation, and velocity information,
    which are essential for robot localization, navigation, and path planning.
    The background task continuously monitors odometry updates to maintain
    accurate robot state information.
    """

    def __init__(self, config: UnitreeGo2OdomConfig):
        """
        Initialize Odom background task with configuration.

        Parameters
        ----------
        config : OdomConfig
            Configuration for the Unitree Go2 Odom background task, including connection parameters and options.
        """
        super().__init__(config)

        unitree_ethernet = self.config.unitree_ethernet
        self.odom_provider = UnitreeGo2OdomProvider(unitree_ethernet)
        logging.info(f"Initialized Unitree Go2 Odom Provider: {unitree_ethernet}")
