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
    Background task for managing Unitree G1 robot location data.

    This background task initializes and manages a UnitreeG1LocationsProvider
    that periodically fetches location information from an HTTP API endpoint.
    The provider runs in a background thread and maintains a thread-safe cache
    of location data that can be accessed by other components of the system.

    The location data is used for navigation, path planning, and location-based
    actions in Unitree G1 robot applications. The provider automatically refreshes
    the location cache at configurable intervals to ensure up-to-date information.
    """

    def __init__(self, config: UnitreeG1LocationsConfig):
        """
        Initialize the Unitree G1 Locations background task.

        Parameters
        ----------
        config : UnitreeG1LocationsConfig
            Configuration object containing:
            - base_url: The HTTP endpoint URL for fetching locations
            - timeout: Request timeout in seconds for HTTP calls
            - refresh_interval: How often to refresh location data in seconds

        Notes
        -----
        The provider is automatically started during initialization and will
        begin fetching location data in the background. The background thread
        runs as a daemon thread and will be terminated when the main process exits.
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
