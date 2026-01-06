import logging

from backgrounds.base import Background, BackgroundConfig
from providers.d435_provider import D435Provider


class D435(Background[BackgroundConfig]):
    """
    Reads depth data from D435 provider and processes it to detect obstacles.
    """

    def __init__(self, config: BackgroundConfig):
        super().__init__(config)

        self.d435_provider = D435Provider()
        self.d435_provider.start()
        logging.info("Initiated D435 Provider in background")
