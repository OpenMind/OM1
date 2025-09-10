import time

# Import the necessary base classes and YOUR existing SpeakInput interface
from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.ub_tts_provider import UbTtsProvider


class UbTtsConnector(ActionConnector[SpeakInput]):
    """
    A "Speak" connector that uses the UbTtsProvider to perform Text-to-Speech.
    This connector is compatible with the standard SpeakInput interface.
    """

    def __init__(self, config: ActionConfig):
        """
        Initializes the connector and its underlying TTS provider.
        """
        super().__init__(config)

        robot_ip = getattr(self.config, "robot_ip", None)
        base_url = getattr(
            self.config, "ub_tts_base_url", f"http://{robot_ip}:9090/v1/"
        )

        # Instantiate the provider with the correct URL
        self.tts = UbTtsProvider(url=f"{base_url}voice/tts")

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Handles the incoming action by passing it to the TTS provider.

        Args:
            output_interface (SpeakInput): The standard speak interface object.
        """
        # Call the provider's speak method using data from SpeakInput.
        # The text comes from the 'action' field.
        # 'interrupt' and 'timestamp' use default values since they are not in SpeakInput.
        self.tts.speak(
            tts=output_interface.action,
            interrupt=True,
            timestamp=int(time.time()),  # Use current time as a sensible default
        )
