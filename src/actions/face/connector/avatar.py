import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.face.interface import FaceInput
from providers.avatar_provider import AvatarProvider


class FaceAvatarConnector(ActionConnector[FaceInput]):
    def __init__(self, config: ActionConfig):
        """
        Initialize the FaceAvatarConnector with AvatarProvider.

        Parameters:
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

        Parameters:
        ----------
        output_interface : FaceInput
        """
        action = output_interface.action

        success = self.avatar_provider.send_avatar_command(action)

        if success:
            logging.info(f"Avatar face command sent: {action}")
        else:
            logging.warning(f"Failed to send avatar face command: {action}")

    def stop(self):
        """
        Stop and cleanup AvatarProvider.
        """
        self.avatar_provider.stop()
        logging.info("AvatarProvider stopped")

    def tick(self) -> None:
        time.sleep(60)
