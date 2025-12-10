import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider


class UnitreeG1Navigation(Background):
    """
    Reads navigation data from UnitreeG1NavigationProvider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)
        self.unitree_g1_navigation_provider = UnitreeG1NavigationProvider()
        self.unitree_g1_navigation_provider.start()
        logging.info("Unitree G1 Navigation Provider initialized in background")
