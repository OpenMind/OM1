import json
import logging
import time
from typing import Optional
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.asr_rtsp_provider import ASRRTSPProvider
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import (
    AudioStatus,
    String,
    TTSStatusRequest,
    TTSStatusResponse,
    open_zenoh_session,
    prepare_header,
)


class SpeakElevenLabsTTSConfig(ActionConfig):
    """
    Configuration for ElevenLabs TTS connector.

    Parameters
    ----------
    elevenlabs_api_key : Optional[str]
        ElevenLabs API key.
    voice_id : str
        ElevenLabs voice ID.
    model_id : str
        ElevenLabs model ID.
    output_format : str
        ElevenLabs output format.
    silence_rate : int
        Number of responses to skip before speaking.
    """

    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs API key",
    )
    voice_id: str = Field(
        default="JBFqnCBsd6RMkjVDRZzb",
        description="ElevenLabs voice ID",
    )
    model_id: str = Field(
        default="eleven_flash_v2_5",
        description="ElevenLabs model ID",
    )
    output_format: str = Field(
        default="mp3_44100_128",
        description="ElevenLabs output format",
    )
    silence_rate: int = Field(
        default=0,
        description="Number of responses to skip before speaking",
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt when ASR detects speech during playback",
    )


# unstable / not released
# from zenoh.ext import HistoryConfig, Miss, RecoveryConfig, declare_advanced_subscriber
class SpeakElevenLabsTTSConnector(
    ActionConnector[SpeakElevenLabsTTSConfig, SpeakInput]
):
    """
    A "Speak" connector that uses the ElevenLabs TTS Provider to perform Text-to-Speech.
    This connector is compatible with the standard SpeakInput interface.
    """

    def __init__(self, config: SpeakElevenLabsTTSConfig):
        """
        Initializes the connector and its underlying TTS provider.

        Parameters
        ----------
        config : SpeakElevenLabsTTSConfig
            Configuration for the connector.
        """
        super().__init__(config)

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Sleep mode configuration
        self.io_provider = IOProvider()
        self.last_voice_command_time = time.time()

        # Eleven Labs TTS configuration
        elevenlabs_api_key = self.config.elevenlabs_api_key
        voice_id = self.config.voice_id
        model_id = self.config.model_id
        output_format = self.config.output_format
        enable_tts_interrupt = self.config.enable_tts_interrupt

        # silence rate
        self.silence_rate = self.config.silence_rate
        self.silence_counter = 0

        # IO Provider
        self.io_provider = IOProvider()

        self.audio_topic = "robot/status/audio"
        self.tts_status_request_topic = "om/tts/request"
        self.tts_status_response_topic = "om/tts/response"
        self.session = None
        self.auido_pub = None

        self.audio_status = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=AudioStatus.STATUS_MIC.UNKNOWN.value,
            status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
            sentence_to_speak=String(""),
        )

        try:
            self.session = open_zenoh_session()
            self.auido_pub = self.session.declare_publisher(self.audio_topic)
            self.session.declare_subscriber(self.audio_topic, self.zenoh_audio_message)
            self.session.declare_subscriber(
                self.tts_status_request_topic, self._zenoh_tts_status_request
            )
            self._zenoh_tts_status_response_pub = self.session.declare_publisher(
                self.tts_status_response_topic
            )

            # Unstable / not released
            # advanced_sub = declare_advanced_subscriber(
            #     self.session,
            #     self.audio_topic,
            #     self.audio_message,
            #     history=HistoryConfig(detect_late_publishers=True),
            #     recovery=RecoveryConfig(heartbeat=True),
            #     subscriber_detection=True,
            # )
            # advanced_sub.sample_miss_listener(self.miss_listener)

            if self.auido_pub:
                self.auido_pub.put(self.audio_status.serialize())

            logging.info("Elevenlabs TTS Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Elevenlabs TTS Zenoh client: {e}")

        # ASR Provider
        base_url = getattr(
            self.config,
            "base_url",
            f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}",
        )
        self.asr = ASRRTSPProvider(ws_url=base_url)

        # Initialize Eleven Labs TTS Provider
        self.tts = ElevenLabsTTSProvider(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            enable_tts_interrupt=enable_tts_interrupt,
        )
        self.tts.start()

        # Configure Eleven Labs TTS Provider to ensure settings are applied
        self.tts.configure(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
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
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Process a speak action by sending text to Elevenlabs TTS.

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

        if self.auido_pub:
            self.auido_pub.put(state.serialize())
            return

        self.tts.register_tts_state_callback(self.asr.audio_stream.on_tts_state_change)
        self.tts.add_pending_message(pending_message)

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

    def stop(self) -> None:
        """
        Stop the Elevenlabs TTS connector and cleanup resources.
        """
        if self.session:
            self.session.close()
            logging.info("Elevenlabs TTS Zenoh client closed")

        if self.asr:
            self.asr.stop()

        if self.tts:
            self.tts.stop()
