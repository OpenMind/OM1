import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider


class UnitreeGo2Locations(Background):
    """
    Reads locations from UnitreeGo2LocationsProvider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        """
        Initialize the Locations background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration for the background task.
        """
        super().__init__(config)

        base_url = getattr(
            self.config,
            "base_url",
            "http://localhost:5000/maps/locations/list",
        )
        timeout = getattr(self.config, "timeout", 5)
        refresh_interval = getattr(self.config, "refresh_interval", 30)

        self.locations_provider = UnitreeGo2LocationsProvider(
            base_url=base_url,
            timeout=timeout,
            refresh_interval=refresh_interval,
        )
        self.locations_provider.start()

        logging.info(
            f"Locations Provider initialized in background (base_url: {base_url}, refresh: {refresh_interval}s)"
        )
