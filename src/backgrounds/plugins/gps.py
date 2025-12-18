import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.gps_provider import GpsProvider


class GpsConfig(BackgroundConfig):
    """
    Configuration for GPS Background.

    Parameters
    ----------
    serial_port : Optional[str]
        Serial port for GPS device.
    """

    serial_port: Optional[str] = Field(
        default=None, description="Serial port for GPS device"
    )


class Gps(Background[GpsConfig]):
    """
    Reads GPS and Magnetometer data from GPS provider.
    """

    def __init__(self, config: GpsConfig):
        super().__init__(config)

        port = self.config.serial_port
        if port is None:
            logging.error("GPS serial port not specified in config")
            return

        self.gps_provider = GpsProvider(serial_port=port)
        logging.info(f"Initiated GPS Provider with serial port: {port} in background")
