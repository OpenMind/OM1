import json
import logging
from uuid import uuid4

from om1_utils import ws
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.move_go2_autonomy.interface import MoveInput


class MoveTronSDKConfig(ActionConfig):
    """
    Configuration class for MoveTronSDKConnector.
    """

    base_url: str = Field(
        default="ws://10.192.1.2:5000",
        description="Base URL for the Tron SDK API",
    )
    accid: str = Field(description="Robot Serial Number")


class MoveTronSDKConnector(ActionConnector[MoveTronSDKConfig, MoveInput]):
    """
    Connector for Move action using the Tron SDK.
    """

    def __init__(self, config: MoveTronSDKConfig):
        """
        Initialize the MoveTronSDK connector.

        Parameters
        ----------
        config : MoveTronSDKConfig
            Configuration object for the connector.
        """
        super().__init__(config)

        self.client = ws.Client(self.config.base_url)
        self.client.start()

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect to the Tron SDK and execute the move command.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol for the action.

        Returns
        -------
        None
            This connector does not return any output.
        """
        logging.info(f"Executing move command: {output_interface.action}")

        x, y, z = 0.0, 0.0, 0.0
        if output_interface.action == "move forwards":
            x = 0.5
        elif output_interface.action == "move back":
            x = -0.5
        elif output_interface.action == "turn left":
            z = 0.5
        elif output_interface.action == "turn right":
            z = -0.5

        if self.client.connected:
            self.client.send_message(
                json.dumps(
                    {
                        "accid": self.config.accid,
                        "title": "request_twist",
                        "guid": str(uuid4()),
                        "data": {"x": x, "y": y, "z": z},
                    }
                )
            )
        else:
            logging.error("Tron webSocket client is not connected.")
