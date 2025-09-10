import logging

from actions.base import ActionConfig, ActionConnector
from actions.face.interface import FaceInput


class FaceAvatarConnector(ActionConnector[FaceInput]):
    """
    Connects face actions to the Avatar provider to send commands.
    """

    def __init__(self, config: ActionConfig):
        """
        Initializes the FaceAvatarConnector with the given configuration.

        Parameters
        ----------
        config : ActionConfig
            The configuration for the action connector.
        """
        super().__init__(config)

    async def connect(self, output_interface: FaceInput) -> None:
        """
        Connects the face action to the Avatar provider and sends the corresponding command.

        Parameters
        ----------
        output_interface : FaceInput
            The output interface containing the face action to be sent.
        """
        new_msg = {"face": ""}

        if output_interface.action == "smile":
            new_msg["face"] = "smile"
        elif output_interface.action == "frown":
            new_msg["face"] = "frown"
        elif output_interface.action == "cry":
            new_msg["face"] = "cry"
        elif output_interface.action == "think":
            new_msg["face"] = "think"
        elif output_interface.action == "joy":
            new_msg["face"] = "joy"
        else:
            logging.info(f"Unknown face type: {output_interface.action}")

        logging.info(f"Sent this to avatar: {new_msg}")
