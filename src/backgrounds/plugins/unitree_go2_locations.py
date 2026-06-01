import logging
from typing import Optional

from pydantic import Field, model_validator

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider


class UnitreeGo2LocationsConfig(BackgroundConfig):
    """
    Configuration for Unitree Go2 Locations Background.

    Parameters
    ----------
    base_url : str
        Base URL for the locations API. If None, determined by use_sim flag.
    use_sim : bool
        Whether to run the connector in the simulator. If true, base_url will be overridden to the simulation endpoint.
    api_key : str
    timeout : int
        Request timeout in seconds.
    refresh_interval : int
        Refresh interval in seconds.
    """

    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the locations API, if None, determined by use_sim flag.",
    )
    use_sim: bool = Field(
        default=False,
        description="Whether to run the connector in the simulator. If true, base_url will be overridden to the simulation endpoint.",
    )
    api_key: str = Field(
        default="",
        description="API key for OpenMind cloud system",
    )
    timeout: int = Field(default=5, description="Request timeout in seconds")
    refresh_interval: int = Field(default=30, description="Refresh interval in seconds")

    @model_validator(mode="after")
    def set_base_url(self) -> "UnitreeGo2LocationsConfig":
        """
        Set base_url based on use_sim if not explicitly provided.

        Returns
        -------
        UnitreeGo2LocationsConfig
            The validated configuration with base_url set if it was None.
        """
        if self.base_url is None:
            if self.use_sim:
                self.base_url = "https://api.openmind.com/api/core/simulation/orchestrator/maps/locations/list"
            else:
                self.base_url = "http://localhost:5000/maps/locations/list"
        return self


class UnitreeGo2Locations(Background[UnitreeGo2LocationsConfig]):
    """
    Reads locations from UnitreeGo2LocationsProvider.
    """

    def __init__(self, config: UnitreeGo2LocationsConfig):
        """
        Initialize the Locations background task.

        Parameters
        ----------
        config : UnitreeGo2LocationsConfig
            Configuration for the background task.
        """
        super().__init__(config)

        base_url = self.config.base_url
        api_key = self.config.api_key
        timeout = self.config.timeout
        refresh_interval = self.config.refresh_interval

        if not base_url:
            logging.error("UnitreeGo2Locations requires a base_url to be set in the configuration")
            return

        self.locations_provider = UnitreeGo2LocationsProvider(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            refresh_interval=refresh_interval,
        )
        self.locations_provider.start()

        logging.info(
            f"Locations Provider initialized in background (base_url: {base_url}, refresh: {refresh_interval}s)"
        )
