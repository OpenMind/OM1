import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.emotion.interface import EmotionInput
from unitree.unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


class EmotionUnitreeConnector(ActionConnector[EmotionInput]):

    def __init__(self, config: ActionConfig):
        super().__init__(config)

        logging.info(f"Emotion system config {config}")

        # create audio_optical client
        self.ao_client = None

        self.unitree_ethernet = getattr(self.config, "unitree_ethernet", None)
        logging.info(f"EmotionUnitreeConnector using ethernet: {self.unitree_ethernet}")

        if self.unitree_ethernet and self.unitree_ethernet != "":
            # ChannelFactoryInitialize(0, self.UNITREE_WIRED_ETHERNET)
            # this can only be done once, at top level
            logging.info(
                f"Emotion system using {self.unitree_ethernet} as the network Ethernet adapter"
            )
            self.ao_client = AudioClient()
            self.ao_client.SetTimeout(10.0)
            self.ao_client.Init()
            self.ao_client.LedControl(0, 255, 0)

    async def connect(self, output_interface: EmotionInput) -> None:

        if not self.ao_client:
            logging.error("No Unitree Emotion Client")
            return

        if output_interface.action == "happy":
            logging.info("Unitree: happy")  # green
            self.ao_client.LedControlNoReply(0, 255, 0)
        elif output_interface.action == "sad":
            logging.info("Unitree: sad")  # yellow
            self.ao_client.LedControlNoReply(255, 255, 0)
        elif output_interface.action == "mad":
            logging.info("Unitree: mad")  # red
            self.ao_client.LedControlNoReply(255, 0, 0)
        elif output_interface.action == "curious":
            logging.info("Unitree: curious")  # blue
            self.ao_client.LedControlNoReply(0, 0, 255)
        else:
            logging.info(f"Unknown emotion: {output_interface.action}")

        logging.info(f"SendThisToUTClient: {output_interface.action}")

    def tick(self) -> None:
        time.sleep(5)
        # logging.info("MoveUnitreeSDKConnector Tick")
