import logging

from actions.base import ActionConfig, ActionConnector
from actions.move_go2_autonomy.interface import MoveInput


class IDELEConnector(ActionConnector[ActionConfig, MoveInput]):
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

        # Register with Prometheus monitor
        self._monitor.register(
            "IDELEConnector",
            metadata={"type": "action", "category": "movement"},
            recovery_callback=None,
        )

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
        self._monitor.heartbeat("IDELEConnector")
        return
