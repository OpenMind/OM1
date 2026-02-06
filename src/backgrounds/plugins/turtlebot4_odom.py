import logging

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.turtlebot4_odom_provider import TurtleBot4OdomProvider


class TurtleBot4OdomConfig(BackgroundConfig):
    """
    Configuration for TurtleBot4 Odom Background.

    Parameters
    ----------
    URID : str
        Unique Robot ID.
    """

    URID: str = Field(default="", description="Unique Robot ID")


class TurtleBot4Odom(Background[TurtleBot4OdomConfig]):
    """
    Background task for reading odometry data from TurtleBot4 provider.

    This background task initializes and manages an TurtleBot4OdomProvider instance
    that reads robot odometry data. The provider can connect via Zenoh
    for distributed robot systems or via Unitree Ethernet for direct
    hardware communication.

    Odometry data includes position, orientation, and velocity information,
    which are essential for robot localization, navigation, and path planning.
    The background task continuously monitors odometry updates to maintain
    accurate robot state information.
    """

    def __init__(self, config: TurtleBot4OdomConfig):
        """
        Initialize Odom background task with configuration.

        Parameters
        ----------
        config : OdomConfig
            Configuration for the TurtleBot4 Odom background task, including connection parameters and options.
        """
        super().__init__(config)

        self.URID = self.config.URID
        self.odom_provider = TurtleBot4OdomProvider(self.URID)
        logging.info(f"Initialized TurtleBot4 Odom Provider with URID: {self.URID}")
