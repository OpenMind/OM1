import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_lidar_localization_provider import (
    UnitreeGo2LidarLocalizationProvider,
)


class UnitreeGo2LidarLocalization(Background[BackgroundConfig]):
    """
    Reads lidar localization data from UnitreeGo2LidarLocalizationProvider.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize the Unitree Go2 Lidar Localization background task.

        This method initializes the UnitreeGo2LidarLocalizationProvider
        and starts it in the background.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background task. The lidar localization provider
            uses default settings and does not require additional configuration
            parameters.
        """
        super().__init__(config)

        self.unitree_go2_lidar_localization_provider = (
            UnitreeGo2LidarLocalizationProvider()
        )
        self.unitree_go2_lidar_localization_provider.start()
        logging.info(
            "Unitree Go2 Lidar Localization Provider initialized in background"
        )
