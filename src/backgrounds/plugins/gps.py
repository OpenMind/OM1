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
    Background task for managing GPS and magnetometer data collection.

    This background task initializes and manages a GpsProvider instance that
    continuously reads GPS positioning data and magnetometer readings from a
    connected GPS device via serial communication. The provider operates at
    115200 baud rate and maintains a persistent connection to the GPS hardware.

    The GPS background enables real-time location tracking and orientation sensing,
    which are essential for autonomous navigation, geofencing, waypoint navigation,
    and location-aware robotics applications. The data collected includes latitude,
    longitude, altitude, and magnetic heading information.

    Typical use cases include:
    - Autonomous outdoor navigation and path planning
    - Real-time location tracking and logging
    - Geofencing and boundary detection
    - Integration with mapping and navigation systems
    - Magnetic heading for orientation awareness

    The GPS provider runs in a separate background thread, continuously polling
    the serial device for new NMEA data and updating the internal state with
    the latest position and magnetometer readings.

    Notes
    -----
    The GPS device must be properly connected to the specified serial port before
    initialization. If the serial port is not specified in the configuration, an
    error will be logged and the provider will not be initialized.
    """

    def __init__(self, config: GpsConfig):
        """
        Initialize GPS background task with configuration.

        Sets up the GPS provider with the specified serial port configuration
        and starts the background data collection process.

        Parameters
        ----------
        config : GpsConfig
            Configuration object containing the GPS settings. The config includes:
            - `serial_port`: Serial port path for the GPS device (e.g., "/dev/ttyUSB0")

        Notes
        -----
        If the serial port is not specified in the configuration, an error will be
        logged and the GPS provider will not be initialized. Ensure the GPS device
        is connected and the serial port has proper read/write permissions.
        """
        super().__init__(config)
        port = self.config.serial_port
        if port is None:
            logging.error("GPS serial port not specified in config")
            return
        self.gps_provider = GpsProvider(serial_port=port)
        logging.info(f"Initiated GPS Provider with serial port: {port} in background")
