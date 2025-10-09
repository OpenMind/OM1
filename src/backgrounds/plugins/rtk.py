import logging

from backgrounds.base import Background, BackgroundConfig
from providers.rtk_provider import RtkProvider


class Rtk(Background):
    """
    Reads RTK data from RTK provider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        port = getattr(config, "serial_port", None)
        if port is None:
            logging.error("RTK serial port not specified in config")
            return

        self.rtk = RtkProvider(serial_port=port)
        logging.info(f"Initiated RTK Provider with serial port: {port} in background")
