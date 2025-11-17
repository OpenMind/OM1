import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput


class SpeakRos2Connector(ActionConnector[SpeakInput]):
    def __init__(self, config: ActionConfig):
        super().__init__(config)

    async def connect(self, output_interface: SpeakInput) -> None:
        new_msg = {"speak": output_interface.action}
        logging.info(f"SendThisToROS2: {new_msg}")
