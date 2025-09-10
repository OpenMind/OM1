import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_state_provider import UnitreeGo2StateProvider


class UnitreeGo2State(Background):
    """
    Reads Unitree Go2 state from UnitreeGo2StateProvider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        unitree_ethernet = getattr(config, "unitree_ethernet", None)
        if not unitree_ethernet:
            logging.error(
                "Unitree Go2 Ethernet channel is not set in the configuration."
            )
            raise ValueError(
                "Unitree Go2 Ethernet channel must be specified in the configuration."
            )

        self.unitree_go2_state_provider = UnitreeGo2StateProvider()
        logging.info("Unitree Go2 State Provider initialized in background")
