import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_location_provider import UnitreeGo2LocationProvider


class UnitreeGo2Location(Background):
    """
    Reads location data from UnitreeGo2LocationProvider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        self.unitree_go2_location_provider = UnitreeGo2LocationProvider()
        self.unitree_go2_location_provider.start()
        logging.info("Unitree Go2 Location Provider initialized in background")
