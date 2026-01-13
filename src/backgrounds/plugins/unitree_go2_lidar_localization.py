import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_lidar_localization_provider import (
    UnitreeGo2LidarLocalizationProvider,
)


class UnitreeGo2LidarLocalization(Background[BackgroundConfig]):
    """
    Background task for reading lidar localization data from Unitree Go2 robot.

    This class manages the integration with UnitreeGo2LidarLocalizationProvider
    to provide real-time lidar-based localization data for navigation and mapping
    applications.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize UnitreeGo2LidarLocalization background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for background task settings.
        """
        super().__init__(config)

        self.unitree_go2_lidar_localization_provider = (
            UnitreeGo2LidarLocalizationProvider()
        )
        self.unitree_go2_lidar_localization_provider.start()
        logging.info(
            "Unitree Go2 Lidar Localization Provider initialized in background"
        )
