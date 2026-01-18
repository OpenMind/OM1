import logging
from typing import Optional

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.asr_provider import ASRProvider
from providers.riva_tts_provider import RivaTTSProvider
from zenoh_msgs import (
    String,
    TTSStatusRequest,
    TTSStatusResponse,
    open_zenoh_session,
    prepare_header,
)


class SpeakRivaTTSConfig(ActionConfig):
    """
    Configuration for Riva TTS connector.

    Parameters
    ----------
    api_key : Optional[str]
        OM API key.
    microphone_device_id : Optional[int]
        Microphone device ID.
    microphone_name : Optional[str]
        Microphone name.
    """

    api_key: Optional[str] = Field(
        default=None,
        description="OM API key",
    )
    microphone_device_id: Optional[int] = Field(
        default=None,
        description="Microphone device ID",
    )
    microphone_name: Optional[str] = Field(
        default=None,
        description="Microphone name",
    )


class SpeakRivaTTSConnector(ActionConnector[SpeakRivaTTSConfig, SpeakInput]):
    """
    A "Speak" connector that uses the RivaTTSProvider to perform Text-to-Speech.
    This connector is compatible with the standard SpeakInput interface.
    """

    def __init__(self, config: SpeakRivaTTSConfig):
        """
        Initializes the connector and its underlying TTS provider.

        Parameters
        ----------
        config : SpeakRivaTTSConfig
            Configuration for the connector.
        """
        super().__init__(config)

        # Get microphone and speaker device IDs and names
        microphone_device_id = self.config.microphone_device_id
        microphone_name = self.config.microphone_name

        # OM API key
        api_key = self.config.api_key

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

            logging.info("Riva TTS Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Riva TTS Zenoh client: {e}")

        # Initialize ASR and TTS providers
        self.asr = ASRProvider(
            ws_url="wss://api-asr.openmind.org",
            device_id=microphone_device_id,
            microphone_name=microphone_name,
        )
        self.tts = RivaTTSProvider(
            url="https://api.openmind.org/api/core/riva/tts",
            api_key=api_key,
        )

        # TTS state
        self.tts_enabled = True

        # Start TTS processing loop
        self.tts.start()

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Process a speak action by sending text to Riva TTS.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.
        """
        # Check if TTS is enabled
        if self.tts_enabled is False:
            logging.info("TTS is disabled, skipping speak action")
            return

        # Block ASR until TTS is done
        self.tts.register_tts_state_callback(self.asr.audio_stream.on_tts_state_change)
        # Add pending message to TTS
        self.tts.add_pending_message(output_interface.action)

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
                status=String(
                    data=("TTS Enabled" if self.tts_enabled else "TTS Disabled")
                ),
            )
            return self._zenoh_tts_status_response_pub.put(
                tts_status_response.serialize()
            )

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
            return self._zenoh_tts_status_response_pub.put(
                ai_status_response.serialize()
            )

        # Disable the TTS
        if code == 0:
            self.tts_enabled = False
            logging.debug("TTS Disabled")
            ai_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=0,
                status=String(data="TTS Disabled"),
            )

            return self._zenoh_tts_status_response_pub.put(
                ai_status_response.serialize()
            )
