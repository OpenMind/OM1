import logging

from actions.base import ActionConfig, ActionConnector
from actions.move_go2_autonomy.interface import MoveInput


class IDLEConnector(ActionConnector[ActionConfig, MoveInput]):
    """
    IDLE connector for Go2 that performs no action.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the IDLE connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration object for the connector.
        """
        super().__init__(config)

    async def connect(self, output_interface: MoveInput) -> None:
        """
        IDLE connector that performs no action.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol for the action. (Not used in this connector)

        Returns
        -------
        None
            This connector does not return any output.
        """
        logging.info("IDLE connector called, doing nothing.")
        return
