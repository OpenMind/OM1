import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.zenoh_publisher_provider import ZenohPublisherProvider


class SpeakZenohConnector(ActionConnector[ActionConfig, SpeakInput]):
    """
    Connector for publishing speak messages via Zenoh.

    When a speak input is received, this connector queues the message
    to be published using the Zenoh publisher provider.
    """

    def __init__(self, config: ActionConfig):
        """
        Initializes the connector and its underlying Zenoh publisher.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the connector.
        """
        super().__init__(config)

        # Determine topic for speech; default to "speech" if not provided.
        speak_topic = getattr(self.config, "speak_topic", None)
        if speak_topic is None:
            speak_topic = "speech"
            # Log the speak_topic being used
            logging.info(
                f"Speak topic not provided. Using default topic: {speak_topic}"
            )
        self.publisher = ZenohPublisherProvider(speak_topic)
        self.publisher.start()

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Queue the speak message for publishing via the Zenoh publisher provider.

        Parameters
        ----------
        output_interface : SpeakInput
            The speak input containing the message (action) to be published.
        """
        self.publisher.add_pending_message(output_interface.action)
