import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider


class UnitreeGo2AMCL(Background[BackgroundConfig]):
    """
    Reads AMCL data from UnitreeGo2AMCLProvider.
    """

    def __init__(self, config: BackgroundConfig):
        super().__init__(config)

        self.unitree_go2_amcl_provider = UnitreeGo2AMCLProvider()
        self.unitree_go2_amcl_provider.start()
        logging.info("Unitree Go2 AMCL Provider initialized in background")
