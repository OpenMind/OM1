import logging

from actions.base import ActionConfig, ActionConnector
from actions.face.interface import FaceInput
from providers.avatar_provider import AvatarProvider


class FaceAvatarConnector(ActionConnector[ActionConfig, FaceInput]):
    """
    Connector to link Face action with AvatarProvider.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the FaceAvatarConnector with AvatarProvider.

        Parameters
        ----------
        config : ActionConfig
            Configuration parameters for the connector.
        """
        super().__init__(config)

        self.avatar_provider = AvatarProvider()
        logging.info("Face system initiated with AvatarProvider")

    async def connect(self, output_interface: FaceInput) -> None:
        """
        Send face command via AvatarProvider.

        Parameters
        ----------
        output_interface : FaceInput
        """
        if output_interface.action == "happy":
            self.avatar_provider.send_avatar_command("Happy")
        elif output_interface.action == "sad":
            self.avatar_provider.send_avatar_command("Sad")
        elif output_interface.action == "curious":
            self.avatar_provider.send_avatar_command("Curious")
        elif output_interface.action == "confused":
            self.avatar_provider.send_avatar_command("Confused")
        elif output_interface.action == "think":
            self.avatar_provider.send_avatar_command("Think")
        elif output_interface.action == "excited":
            self.avatar_provider.send_avatar_command("Excited")
        else:
            logging.warning("Failed to send avatar face command")

        logging.info(f"Avatar face command sent: {output_interface.action}")

    def stop(self):
        """
        Stop and cleanup AvatarProvider.
        """
        self.avatar_provider.stop()
        logging.info("AvatarProvider stopped")
