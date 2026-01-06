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
    Reads RTK data from RTK provider.
    """

    def __init__(self, config: RtkConfig):
        super().__init__(config)

        port = self.config.serial_port
        if port is None:
            logging.error("RTK serial port not specified in config")
            return

        self.rtk = RtkProvider(serial_port=port)
        logging.info(f"Initiated RTK Provider with serial port: {port} in background")
