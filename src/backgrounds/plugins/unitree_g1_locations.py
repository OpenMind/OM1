import logging

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_g1_locations_provider import UnitreeG1LocationsProvider


class UnitreeG1LocationsConfig(BackgroundConfig):
    """
    Configuration for Unitree G1 Locations Background.

    Parameters
    ----------
    base_url : str
        Base URL for the locations API.
    timeout : int
        Request timeout in seconds.
    refresh_interval : int
        Refresh interval in seconds.
    """

    base_url: str = Field(
        default="http://localhost:5000/maps/locations/list",
        description="Base URL for the locations API",
    )
    timeout: int = Field(default=5, description="Request timeout in seconds")
    refresh_interval: int = Field(default=30, description="Refresh interval in seconds")


class UnitreeG1Locations(Background[UnitreeG1LocationsConfig]):
    """
    Background task for fetching and caching location data for Unitree G1 robot.

    This background task periodically fetches location data from an HTTP API endpoint
    using the UnitreeG1LocationsProvider. The location data is used by the robot's
    navigation system to provide available destinations for autonomous navigation.

    The provider runs in a background thread and automatically refreshes the location
    cache at the configured interval, ensuring the robot always has up-to-date
    location information for navigation commands.
    """

    def __init__(self, config: UnitreeG1LocationsConfig):
        """
        Initialize the Unitree G1 Locations background task.

        Parameters
        ----------
        config : UnitreeG1LocationsConfig
            Configuration object containing API endpoint, timeout, and refresh interval
            settings. The config object must include:
            - base_url: HTTP endpoint for the locations API
            - timeout: Request timeout in seconds
            - refresh_interval: How often to refresh locations in seconds
        """
        super().__init__(config)

        base_url = self.config.base_url
        timeout = self.config.timeout
        refresh_interval = self.config.refresh_interval

        self.locations_provider = UnitreeG1LocationsProvider(
            base_url=base_url,
            timeout=timeout,
            refresh_interval=refresh_interval,
        )
        self.locations_provider.start()
        logging.info(
            f"G1 Locations Provider initialized in background (base_url: {base_url}, refresh: {refresh_interval}s)"
        )
