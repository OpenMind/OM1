import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.odom_provider import OdomProvider


class OdomConfig(BackgroundConfig):
    """
    Configuration for Odom Background.

    Parameters
    ----------
    use_zenoh : bool
        Whether to use Zenoh.
    URID : str
        Unique Robot ID.
    unitree_ethernet : Optional[str]
        Unitree Ethernet channel.
    """

    use_zenoh: bool = Field(default=False, description="Whether to use Zenoh")
    URID: str = Field(default="", description="Unique Robot ID")
    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Ethernet channel"
    )


class Odom(Background[OdomConfig]):
    """
    Background task for reading odometry data from Odom provider.

    This background task initializes and manages an OdomProvider instance
    that reads robot odometry data. The provider can connect via Zenoh
    for distributed robot systems or via Unitree Ethernet for direct
    hardware communication.

    Odometry data includes position, orientation, and velocity information,
    which are essential for robot localization, navigation, and path planning.
    The background task continuously monitors odometry updates to maintain
    accurate robot state information.
    """

    def __init__(self, config: OdomConfig):
        """
        Initialize Odom background task with configuration.

        Parameters
        ----------
        config : OdomConfig
            Configuration object for the odometry background task. The configuration
            specifies the connection method (Zenoh or Unitree Ethernet) and
            required connection parameters (URID for Zenoh, ethernet channel for Unitree).
            If use_zenoh is True, the provider will subscribe to odometry data via Zenoh
            using the specified URID. If use_zenoh is False and unitree_ethernet is provided,
            the provider will connect directly to Unitree hardware via Ethernet.
        """
        super().__init__(config)

        use_zenoh = self.config.use_zenoh
        self.URID = self.config.URID
        unitree_ethernet = self.config.unitree_ethernet
        self.odom_provider = OdomProvider(self.URID, use_zenoh, unitree_ethernet)
        if use_zenoh:
            logging.info(f"Odom using Zenoh with URID: {self.URID} in background")
        else:
            logging.info("Odom provider initialized without Zenoh in Odom background")
