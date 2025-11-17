import logging
import time

import zenoh

# Import the necessary base classes and YOUR existing SpeakInput interface
from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.ub_tts_provider import UbTtsProvider
from zenoh_msgs import (
    String,
    TTSStatusRequest,
    TTSStatusResponse,
    open_zenoh_session,
    prepare_header,
)


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
        base_url = getattr(self.config, "ub_tts_base_url", f"http://{robot_ip}:9090/v1/")

        # Zenoh topics
        self.tts_status_request_topic = "om/tts/request"
        self.tts_status_response_topic = "om/tts/response"

        self.session = None

        try:
            self.session = open_zenoh_session()
            self.session.declare_subscriber(
                self.tts_status_request_topic, self._zenoh_tts_status_request
            )
            self._zenoh_tts_status_response_pub = self.session.declare_publisher(
                self.tts_status_response_topic
            )

            logging.info("UB TTS Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening UB TTS Zenoh client: {e}")

        # Instantiate the provider with the correct URL
        self.tts = UbTtsProvider(url=f"{base_url}voice/tts")

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Handles the incoming action by passing it to the TTS provider.

        Parameters
        ----------
        output_interface : SpeakInput
            The input protocol containing the action details.
        """
        if not self.tts_enabled:
            logging.warning("TTS is currently disabled. Ignoring speak request.")
            return

        # Call the provider's speak method using data from SpeakInput.
        # The text comes from the 'action' field.
        # 'interrupt' and 'timestamp' use default values since they are not in SpeakInput.
        self.tts.speak(
            tts=output_interface.action,
            interrupt=True,
            timestamp=int(time.time()),  # Use current time as a sensible default
        )

    def _zenoh_tts_status_request(self, data: zenoh.Sample):
        """
        Process an incoming TTS control status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        tts_status = TTSStatusRequest.deserialize(data.payload.to_bytes())
        logging.debug(f"Received TTS Control Status message: {tts_status}")

        code = tts_status.code
        request_id = tts_status.request_id

        # Read the current status
        if code == 2:
            tts_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=1 if self.tts_enabled else 0,
                status=String(data=("TTS Enabled" if self.tts_enabled else "TTS Disabled")),
            )
            return self._zenoh_tts_status_response_pub.put(tts_status_response.serialize())

        # Enable the TTS
        if code == 1:
            self.tts_enabled = True
            logging.debug("TTS Enabled")

            ai_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=1,
                status=String(data="TTS Enabled"),
            )
            return self._zenoh_tts_status_response_pub.put(ai_status_response.serialize())

        # Disable the TTS
        if code == 0:
            self.tts_enabled = False
            logging.info("TTS Disabled")
            ai_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=0,
                status=String(data="TTS Disabled"),
            )

            return self._zenoh_tts_status_response_pub.put(ai_status_response.serialize())
