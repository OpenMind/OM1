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
    Background task for reading GPS and magnetometer data.

    This background task initializes and manages a GpsProvider instance
    that connects to a GPS device via serial port. The provider reads
    GPS coordinates (latitude, longitude, altitude) and magnetometer
    data (yaw angle) from the connected device.

    The GPS data is used for robot localization and navigation, providing
    absolute position information in outdoor environments where GPS signals
    are available.
    """

    def __init__(self, config: GpsConfig):
        """
        Initialize GPS background task with configuration.

        Parameters
        ----------
        config : GpsConfig
            Configuration object for the GPS background task. Must include
            the serial_port parameter specifying the serial port where the
            GPS device is connected. If serial_port is None, the provider
            will not be initialized and an error will be logged.
        """
        super().__init__(config)

        port = self.config.serial_port
        if port is None:
            logging.error("GPS serial port not specified in config")
            return

        self.gps_provider = GpsProvider(serial_port=port)
        logging.info(f"Initiated GPS Provider with serial port: {port} in background")
