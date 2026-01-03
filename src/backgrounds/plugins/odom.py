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
    Reads odometry data from Odom provider.
    """

    def __init__(self, config: OdomConfig):
        super().__init__(config)

        use_zenoh = self.config.use_zenoh
        self.URID = self.config.URID
        unitree_ethernet = self.config.unitree_ethernet
        self.odom_provider = OdomProvider(self.URID, use_zenoh, unitree_ethernet)
        if use_zenoh:
            logging.info(f"Odom using Zenoh with URID: {self.URID} in background")
        else:
            logging.info("Odom provider initialized without Zenoh in Odom background")
