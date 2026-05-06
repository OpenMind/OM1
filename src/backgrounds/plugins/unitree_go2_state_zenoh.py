import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_state_zenoh_provider import UnitreeGo2StateZenohProvider


class UnitreeGo2StateZenohConfig(BackgroundConfig):
    """
    Configuration for the Unitree Go2 State Zenoh background.

    Parameters
    ----------
    api_key : Optional[str]
        API Key for Zenoh session, if required.
    use_sim : bool
        Whether to use the simulation Zenoh endpoint instead of a local one.
    """

    api_key: Optional[str] = Field(default=None, description="API Key for Zenoh session, if required.")
    use_sim: bool = Field(
        default=False,
        description="Whether to use the simulation Zenoh endpoint instead of a local one.",
    )


class UnitreeGo2StateZenoh(Background[UnitreeGo2StateZenohConfig]):
    """Background that subscribes to the Unitree Go2 SportModeState over Zenoh."""

    def __init__(self, config: UnitreeGo2StateZenohConfig):
        """
        Initialize the background and start the Zenoh subscriber.

        Parameters
        ----------
        config : UnitreeGo2StateZenohConfig
            Configuration for the background.
        """
        super().__init__(config)

        self.unitree_go2_state_provider = UnitreeGo2StateZenohProvider(self.config.api_key, self.config.use_sim)
        logging.info("Unitree Go2 State Zenoh Provider initialized in background")
