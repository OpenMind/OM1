import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput


feat/gazebo-improve-bounty-363
class SpeakRos2Connector(ActionConnector[SpeakInput]):

class SpeakRos2Connector(ActionConnector[ActionConfig, SpeakInput]):
    """
    A "Speak" connector that sends speak commands to a ROS2 system.
    This connector is compatible with the standard SpeakInput interface.
    """

main
    def __init__(self, config: ActionConfig):
        """
        Initializes the connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the connector.
        """
        super().__init__(config)

    async def connect(self, output_interface: SpeakInput) -> None:
feat/gazebo-improve-bounty-363

        """
        Process a speak action by sending text to ROS2.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.
        """
main
        new_msg = {"speak": output_interface.action}
        logging.info(f"SendThisToROS2: {new_msg}")
