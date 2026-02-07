import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_g1_odom_provider import UnitreeG1OdomProvider


class UnitreeG1OdomConfig(BackgroundConfig):
    """
    Configuration for Unitree G1 Odom Background.

    Parameters
    ----------
    unitree_ethernet : Optional[str]
        Unitree Ethernet channel.
    """

    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Ethernet channel"
    )


class UnitreeG1Odom(Background[UnitreeG1OdomConfig]):
    """
    Background task for reading odometry data from Unitree G1 provider.

    This background task initializes and manages an UnitreeG1OdomProvider instance
    that reads robot odometry data. The provider can connect via Zenoh
    for distributed robot systems or via Unitree Ethernet for direct
    hardware communication.

    Odometry data includes position, orientation, and velocity information,
    which are essential for robot localization, navigation, and path planning.
    The background task continuously monitors odometry updates to maintain
    accurate robot state information.
    """

    def __init__(self, config: UnitreeG1OdomConfig):
        """
        Initialize Odom background task with configuration.

        Parameters
        ----------
        config : OdomConfig
            Configuration for the Unitree G1 Odom background task, including connection parameters and options.
        """
        super().__init__(config)

        unitree_ethernet = self.config.unitree_ethernet
        self.odom_provider = UnitreeG1OdomProvider(unitree_ethernet)
        logging.info(f"Initialized Unitree G1 Odom Provider: {unitree_ethernet}")
