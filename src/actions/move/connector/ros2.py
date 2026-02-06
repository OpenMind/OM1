import logging

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput


class MoveUnitreeSDKConnector(ActionConnector[ActionConfig, MoveInput]):
    """
    Connector to link Move action with Unitree SDK via ROS2.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the MoveUnitreeSDKConnector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect the input protocol to the move action via Unitree SDK.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol containing the action details.
        """
        new_msg = {"move": ""}

        # stub to show how to do this
        if output_interface.action == "stand still":
            new_msg["move"] = "stand still"
        elif output_interface.action == "sit":
            new_msg["move"] = "sit"
        elif output_interface.action == "dance":
            new_msg["move"] = "dance"
        elif output_interface.action == "shake paw":
            new_msg["move"] = "shake paw"
        elif output_interface.action == "walk":
            new_msg["move"] = "walk"
        elif output_interface.action == "walk back":
            new_msg["move"] = "walk back"
        elif output_interface.action == "run":
            new_msg["move"] = "run"
        elif output_interface.action == "jump":
            new_msg["move"] = "jump"
        elif output_interface.action == "wag tail":
            new_msg["move"] = "wag tail"
        else:
            logging.info(f"Other move type: {output_interface.action}")
            # raise ValueError(f"Unknown move type: {output_interface.action}")

        logging.info(f"SendThisToROS2: {new_msg}")
