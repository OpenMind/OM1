import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider


class UnitreeGo2Navigation(Background[BackgroundConfig]):
    """
    Reads navigation data from UnitreeGo2NavigationProvider.
    """

    def __init__(self, config: BackgroundConfig):
        super().__init__(config)

        self.unitree_go2_navigation_provider = UnitreeGo2NavigationProvider()
        self.unitree_go2_navigation_provider.start()
        logging.info("Unitree Go2 Navigation Provider initialized in background")
