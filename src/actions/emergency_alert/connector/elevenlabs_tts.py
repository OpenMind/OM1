import json
import logging
import time
from typing import Optional
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.emergency_alert.interface import EmergencyAlertInput
from providers.asr_rtsp_provider import ASRRTSPProvider
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import (
    AudioStatus,
    String,
    TTSStatusRequest,
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


# unstable / not released
# from zenoh.ext import HistoryConfig, Miss, RecoveryConfig, declare_advanced_subscriber
class EmergencyAlertElevenLabsTTSConnector(
    ActionConnector[SpeakElevenLabsTTSConfig, EmergencyAlertInput]
):
    """
    Connector that uses ElevenLabs TTS for emergency alerts.
    """

    def __init__(self, config: SpeakElevenLabsTTSConfig):
        """
        Initialize the EmergencyAlertElevenLabsTTSConnector.

        Parameters
        ----------
        config : SpeakElevenLabsTTSConfig
            Configuration for the action connector.
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

        # IO Provider
        self.io_provider = IOProvider()

        self.audio_topic = "robot/status/audio"
        self.tts_status_request_topic = "om/tts/request"
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

        base_url = getattr(
            self.config,
            "base_url",
            f"wss://api.openmind.org/api/core/google/asr?api_key={api_key}",
        )
        self.asr = ASRRTSPProvider(ws_url=base_url)

        self.tts = ElevenLabsTTSProvider(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        self.tts.start()

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

    async def connect(self, output_interface: EmergencyAlertInput) -> None:
        """
        Connect the input protocol to the ElevenLabs TTS action for emergency alerts.

        Parameters
        ----------
        output_interface : EmergencyAlertInput
            The input protocol containing the action details.
        """
        if self.tts_enabled is False:
            logging.info("TTS is disabled, skipping TTS action")
            return

        # Add pending message to TTS
        pending_message = self.tts.create_pending_message(output_interface.action)

        # Store robot message to conversation history only if there was ASR input
        if (
            self.io_provider.llm_prompt is not None
            and "INPUT: Voice" in self.io_provider.llm_prompt
        ):
            self.conversation_provider.store_robot_message(output_interface.action)

        # Avoid queuing too many TTS messages
        if self.tts.get_pending_message_count() > 0:
            logging.warning(
                "Too many pending TTS messages, skipping adding new message"
            )
            return

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

        # Enable the TTS
        if code == 1:
            self.tts_enabled = True
            logging.debug("TTS Enabled")

        # Disable the TTS
        if code == 0:
            self.tts_enabled = False
            logging.debug("TTS Disabled")
