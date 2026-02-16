import logging

from actions.base import ActionConfig, ActionConnector
from actions.recover_stand.interface import RecoverAction, RecoverInput

try:
    from unitree.unitree_sdk2py.go2.sport.sport_client import (
        SportClient,  # type: ignore
    )
except ImportError:
    SportClient = None  # type: ignore[assignment, misc]
    logging.warning(
        "Unitree SDK not found. RecoverStand connector will not be able to send commands."
    )


class RecoverStandUnitreeConfig(ActionConfig):
    """
    Configuration for RecoverStand Unitree Go2 connector.
    """

    pass


class RecoverStandUnitreeConnector(
    ActionConnector[RecoverStandUnitreeConfig, RecoverInput]
):
    """
    Connector that sends RecoveryStand commands to Unitree Go2 via SportClient.
    """

    def __init__(self, config: RecoverStandUnitreeConfig):
        """
        Initialize the RecoverStand connector with SportClient.

        Parameters
        ----------
        config : RecoverStandUnitreeConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        self.sport_client = None
        try:
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
            logging.info("RecoverStand Unitree sport client initialized")
        except Exception as e:
            logging.error(f"Error initializing RecoverStand sport client: {e}")

    async def connect(self, output_interface: RecoverInput) -> None:
        """
        Execute the recovery stand command.

        Parameters
        ----------
        output_interface : RecoverInput
            The input containing the recovery action.
        """
        action = output_interface.action
        logging.info(f"RecoverStandUnitreeConnector received action: {action}")

        if action == RecoverAction.RECOVER:
            if self.sport_client is not None:
                try:
                    logging.info("Executing RecoveryStand command")
                    self.sport_client.RecoveryStand()
                except Exception as e:
                    logging.error(f"Error executing RecoveryStand: {e}")
            else:
                logging.error(
                    "Cannot execute RecoveryStand: sport client not initialized"
                )
        else:
            logging.warning(f"Unknown recover action: {action}")
