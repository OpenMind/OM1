import logging

from backgrounds.base import Background, BackgroundConfig
from providers.gps_provider import GpsProvider


class Gps(Background):
    """
    Reads GPS and Magnetometer data from GPS provider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        port = getattr(config, "serial_port", None)
        if port is None:
            logging.error("GPS serial port not specified in config")
            return

        self.gps_provider = GpsProvider(serial_port=port)
        logging.info(f"Initiated GPS Provider with serial port: {port} in background")
