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
    Background task for managing Real-Time Kinematic (RTK) GPS data collection.

    This background task initializes and manages an RtkProvider instance that
    continuously reads high-precision RTK GPS positioning data from a connected
    RTK receiver via serial communication. The provider uses the pynmeagps library
    to parse NMEA messages and operates at 115200 baud rate with a 0.2-second
    timeout for responsive data acquisition.

    RTK GPS provides centimeter-level positioning accuracy by using correction data
    from a base station, making it significantly more precise than standard GPS.
    This enhanced accuracy is crucial for applications requiring precise navigation,
    surveying, or autonomous operations in constrained environments.

    Typical use cases include:
    - High-precision autonomous navigation and path following
    - Agricultural robotics requiring accurate field positioning
    - Construction and surveying applications
    - Precision landing and docking maneuvers
    - Accurate waypoint tracking and geofencing

    The RTK provider runs in a separate background thread, continuously reading
    and parsing NMEA messages from the serial device to maintain up-to-date
    position information with centimeter-level accuracy.

    Notes
    -----
    The RTK receiver must be properly connected to the specified serial port and
    configured to receive correction data (either via NTRIP, radio, or other means)
    before initialization. Without correction data, the RTK receiver will fall back
    to standard GPS accuracy. If the serial port is not specified in the
    configuration, an error will be logged and the provider will not be initialized.
    """

    def __init__(self, config: RtkConfig):
        """
        Initialize RTK background task with configuration.

        Sets up the RTK provider with the specified serial port configuration
        and starts the background high-precision positioning data collection process.

        Parameters
        ----------
        config : RtkConfig
            Configuration object containing the RTK settings. The config includes:
            - `serial_port`: Serial port path for the RTK device (e.g., "/dev/ttyUSB0")

        Notes
        -----
        If the serial port is not specified in the configuration, an error will be
        logged and the RTK provider will not be initialized. Ensure the RTK receiver
        is connected, the serial port has proper read/write permissions, and the
        device is receiving correction data for optimal accuracy.
        """
        super().__init__(config)
        port = self.config.serial_port
        if port is None:
            logging.error("RTK serial port not specified in config")
            return
        self.rtk = RtkProvider(serial_port=port)
        logging.info(f"Initiated RTK Provider with serial port: {port} in background")
