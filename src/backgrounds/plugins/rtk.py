import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.rtk_provider import RtkProvider


class RtkConfig(BackgroundConfig):
    """
    Configuration for RTK Background.

    Parameters
    ----------
    serial_port : Optional[str]
        Serial port for RTK device.
    """

    serial_port: Optional[str] = Field(
        default=None, description="Serial port for RTK device"
    )


class Rtk(Background[RtkConfig]):
    """
    Background task for reading RTK (Real-Time Kinematic) positioning data.

    This background task initializes and manages a RtkProvider instance
    that connects to an RTK GPS device via serial port. RTK technology provides
    centimeter-level positioning accuracy by using carrier-phase measurements
    and corrections from a reference station.

    The RTK data is used for high-precision robot localization and navigation,
    particularly in applications requiring accurate positioning such as
    autonomous navigation, mapping, and precision agriculture.
    """

    def __init__(self, config: RtkConfig):
        """
        Initialize RTK background task with configuration.

        Parameters
        ----------
        config : RtkConfig
            Configuration object containing RTK settings.
        """
        super().__init__(config)

        port = self.config.serial_port
        if port is None:
            logging.error("RTK serial port not specified in config")
            return

        self.rtk = RtkProvider(serial_port=port)
        logging.info(f"Initiated RTK Provider with serial port: {port} in background")
