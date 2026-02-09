import json
import logging
import time
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.io_provider import IOProvider
from providers.kokoro_tts_provider import KokoroTTSProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import (
    AudioStatus,
    String,
    TTSStatusRequest,
    TTSStatusResponse,
    open_zenoh_session,
    prepare_header,
)


class SpeakKokoroTTSConfig(ActionConfig):
    """
    Configuration for Kokoro TTS connector.

    Parameters
    ----------
    voice_id : str
        Kokoro voice ID.
    model_id : str
        Kokoro model ID.
    output_format : str
        Kokoro output format.
    rate : int
        Audio sample rate in Hz.
    enable_tts_interrupt : bool
        Enable TTS interrupt when ASR detects speech during playback.
    silence_rate : int
        Number of responses to skip before speaking.
    """

    voice_id: str = Field(
        default="af_bella",
        description="Kokoro voice ID",
    )
    model_id: str = Field(
        default="kokoro",
        description="Kokoro model ID",
    )
    output_format: str = Field(
        default="pcm",
        description="Kokoro output format",
    )
    rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz",
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt when ASR detects speech during playback",
    )
    silence_rate: int = Field(
        default=0,
        description="Number of responses to skip before speaking",
    )


class SpeakKokoroTTSConnector(ActionConnector[SpeakKokoroTTSConfig, SpeakInput]):
    """
    A "Speak" connector that uses the Kokoro TTS Provider to perform Text-to-Speech.
    This connector is compatible with the standard SpeakInput interface.
    """

    def __init__(self, config: SpeakKokoroTTSConfig):
        """
        Initializes the connector and its underlying TTS provider.

        Parameters
        ----------
        config : SpeakKokoroTTSConfig
            Configuration for the connector.
        """
        super().__init__(config)

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Sleep mode configuration
        self.last_voice_command_time = time.time()

        # silence rate
        self.silence_rate = self.config.silence_rate
        self.silence_counter = 0

        # IO Provider
        self.io_provider = IOProvider()

        # Zenoh configuration
        self.audio_topic = "robot/status/audio"
        self.tts_status_request_topic = "om/tts/request"
        self.tts_status_response_topic = "om/tts/response"
        self.session = None
        self.audio_pub = None
        self._zenoh_tts_status_response_pub = None

        self.audio_status = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=AudioStatus.STATUS_MIC.UNKNOWN.value,
            status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
            sentence_to_speak=String(""),
        )

        # Initialize Zenoh session
        try:
            self.session = open_zenoh_session()
            self.audio_pub = self.session.declare_publisher(self.audio_topic)
            self.session.declare_subscriber(self.audio_topic, self.zenoh_audio_message)
            self.session.declare_subscriber(
                self.tts_status_request_topic, self._zenoh_tts_status_request
            )
            self._zenoh_tts_status_response_pub = self.session.declare_publisher(
                self.tts_status_response_topic
            )

            if self.audio_pub:
                self.audio_pub.put(self.audio_status.serialize())

            logging.info("Kokoro TTS Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Kokoro TTS Zenoh client: {e}")

        # Kokoro TTS configuration
        voice_id = self.config.voice_id
        model_id = self.config.model_id
        output_format = self.config.output_format
        rate = self.config.rate
        enable_tts_interrupt = self.config.enable_tts_interrupt

        # Initialize Kokoro TTS Provider
        self.tts = KokoroTTSProvider(
            url="http://127.0.0.1:8880/v1",
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            rate=rate,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.tts.start()

        # Configure Kokoro TTS Provider to ensure settings are applied
        self.tts.configure(
            url="http://127.0.0.1:8880/v1",
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            rate=rate,
            enable_tts_interrupt=enable_tts_interrupt,
        )

        # TTS status
        self.tts_enabled = True

        # Initialize conversation provider
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

    def zenoh_audio_message(self, data: zenoh.Sample):
        """
        Process an incoming audio status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        try:
            self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())
        except Exception as e:
            logging.error(f"Error deserializing audio status: {e}")

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Process a speak action by sending text to Kokoro TTS.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.
        """
        if self.tts_enabled is False:
            logging.info("TTS is disabled, skipping TTS action")
            return

        if (
            self.silence_rate > 0
            and self.silence_counter < self.silence_rate
            and self.io_provider.llm_prompt is not None
            and "INPUT: Voice" not in self.io_provider.llm_prompt
        ):
            self.silence_counter += 1
            logging.info(
                f"Skipping TTS due to silence_rate {self.silence_rate}, counter {self.silence_counter}"
            )
            return

        self.silence_counter = 0

        # Add pending message to TTS
        pending_message = self.tts.create_pending_message(output_interface.action)

        # Store robot message to conversation history only if there was ASR input
        if (
            self.io_provider.llm_prompt is not None
            and "INPUT: Voice" in self.io_provider.llm_prompt
        ):
            self.conversation_provider.store_robot_message(output_interface.action)

        state = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=self.audio_status.status_mic,
            status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
            sentence_to_speak=String(json.dumps(pending_message)),
        )

        if self.audio_pub:
            self.audio_pub.put(state.serialize())
            return

        self.tts.add_pending_message(pending_message)

    def _zenoh_tts_status_request(self, data: zenoh.Sample):
        """
        Process an incoming TTS control status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        try:
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
                if self._zenoh_tts_status_response_pub:
                    self._zenoh_tts_status_response_pub.put(
                        tts_status_response.serialize()
                    )
                return

            # Enable the TTS
            if code == 1:
                self.tts_enabled = True
                logging.debug("TTS Enabled")

                tts_status_response = TTSStatusResponse(
                    header=prepare_header(tts_status.header.frame_id),
                    request_id=request_id,
                    code=1,
                    status=String(data="TTS Enabled"),
                )
                if self._zenoh_tts_status_response_pub:
                    self._zenoh_tts_status_response_pub.put(
                        tts_status_response.serialize()
                    )
                return

            # Disable the TTS
            if code == 0:
                self.tts_enabled = False
                logging.debug("TTS Disabled")
                tts_status_response = TTSStatusResponse(
                    header=prepare_header(tts_status.header.frame_id),
                    request_id=request_id,
                    code=0,
                    status=String(data="TTS Disabled"),
                )
                if self._zenoh_tts_status_response_pub:
                    self._zenoh_tts_status_response_pub.put(
                        tts_status_response.serialize()
                    )
                return

        except Exception as e:
            logging.error(f"Error processing TTS status request: {e}")

    def stop(self) -> None:
        """
        Stop the Kokoro TTS connector and cleanup resources.
        """
        if self.session:
            self.session.close()
            logging.info("Kokoro TTS Zenoh client closed")

        if self.tts:
            self.tts.stop()
