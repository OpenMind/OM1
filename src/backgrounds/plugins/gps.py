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
    Background task for reading GPS position and magnetometer heading data.

    Manages a GpsProvider instance that connects to a GPS device via serial port,
    processing GPS location data (latitude, longitude, altitude) and magnetometer
    data (compass heading) for outdoor navigation and orientation.

    GPS data provides global position information for outdoor localization, while
    magnetometer data provides heading information to determine orientation.
    """

    def __init__(self, config: GpsConfig):
        """
        Initialize the Gps background task instance.

        Sets up the GPS provider using the specified serial port from the configuration.

        Parameters
        ----------
        config : GpsConfig
            Configuration object containing the GPS serial port.
        """
        super().__init__(config)

        port = self.config.serial_port
        if port is None:
            logging.error("GPS serial port not specified in config")
            return

        self.gps_provider = GpsProvider(serial_port=port)
        logging.info(f"Initiated GPS Provider with serial port: {port} in background")
