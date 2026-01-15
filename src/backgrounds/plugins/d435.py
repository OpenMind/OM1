import logging

from backgrounds.base import Background, BackgroundConfig
from providers.d435_provider import D435Provider


class D435(Background[BackgroundConfig]):
    """
    Background task for reading depth data from D435 camera.

    This background task initializes and manages a D435Provider instance
    that subscribes to depth camera data via Zenoh. The provider processes
    depth information to detect obstacles in the robot's environment.

    The D435 camera provides depth sensing capabilities, which are used
    for obstacle detection and navigation assistance.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize D435 background task with configuration.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background task. The base configuration
            is used as D435 does not require additional parameters.
        """
        super().__init__(config)

        self.d435_provider = D435Provider()
        self.d435_provider.start()
        logging.info("Initiated D435 Provider in background")
