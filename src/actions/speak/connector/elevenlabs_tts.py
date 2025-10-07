import json
import logging
import time
from uuid import uuid4

import zenoh

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from providers.asr_provider import ASRProvider
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.io_provider import IOProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import AudioStatus, String, open_zenoh_session, prepare_header


# unstable / not released
# from zenoh.ext import HistoryConfig, Miss, RecoveryConfig, declare_advanced_subscriber
class SpeakElevenLabsTTSConnector(ActionConnector[SpeakInput]):

    def __init__(self, config: ActionConfig):

        super().__init__(config)

        # Get microphone and speaker device IDs and names
        microphone_device_id = getattr(self.config, "microphone_device_id", None)
        microphone_name = getattr(self.config, "microphone_name", None)

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Sleep mode configuration
        self.io_provider = IOProvider()
        self.last_voice_command_time = time.time()
        self.auto_sleep_mode = getattr(config, "auto_sleep_mode", True)
        self.auto_sleep_time = getattr(config, "auto_sleep_time", 300)

        # Eleven Labs TTS configuration
        elevenlabs_api_key = getattr(self.config, "elevenlabs_api_key", None)
        voice_id = getattr(self.config, "voice_id", "JBFqnCBsd6RMkjVDRZzb")
        model_id = getattr(self.config, "model_id", "eleven_flash_v2_5")
        output_format = getattr(self.config, "output_format", "mp3_44100_128")

        # silence rate
        self.silence_rate = getattr(self.config, "silence_rate", 0)
        self.silence_counter = 0

        # IO Provider
        self.io_provider = IOProvider()

        self.topic = "robot/status/audio"
        self.session = None
        self.pub = None

        self.audio_status = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=AudioStatus.STATUS_MIC.UNKNOWN.value,
            status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
            sentence_to_speak=String(""),
        )

        try:
            self.session = open_zenoh_session()
            self.pub = self.session.declare_publisher(self.topic)
            self.session.declare_subscriber(self.topic, self.zenoh_audio_message)

            # Unstable / not released
            # advanced_sub = declare_advanced_subscriber(
            #     self.session,
            #     self.topic,
            #     self.audio_message,
            #     history=HistoryConfig(detect_late_publishers=True),
            #     recovery=RecoveryConfig(heartbeat=True),
            #     subscriber_detection=True,
            # )
            # advanced_sub.sample_miss_listener(self.miss_listener)

            if self.pub:
                self.pub.put(self.audio_status.serialize())

            logging.info("TTS Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening TTS Zenoh client: {e}")

        # Initialize ASR and TTS providers
        self.asr = ASRProvider(
            ws_url="wss://api-asr.openmind.org",
            device_id=microphone_device_id,
            microphone_name=microphone_name,
        )

        self.tts = ElevenLabsTTSProvider(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        self.tts.start()
        self.tts.add_pending_message("Woof Woof")

        # Initialize conversation provider
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

    def zenoh_audio_message(self, data: zenoh.Sample):
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

    async def connect(self, output_interface: SpeakInput) -> None:
        if (
            self.silence_rate > 0
            and self.silence_counter < self.silence_rate
            and "INPUT: Voice" not in self.io_provider.llm_prompt
        ):
            self.silence_counter += 1
            logging.info(
                f"Skipping TTS due to silence_rate {self.silence_rate}, counter {self.silence_counter}"
            )
            return

        self.silence_counter = 0

        if self.auto_sleep_mode:
            voice_input = self.io_provider.inputs.get("Voice")
            if voice_input:
                self.last_voice_command_time = voice_input.timestamp

            if time.time() - self.last_voice_command_time > self.auto_sleep_time:
                return

        # Add pending message to TTS
        pending_message = self.tts.create_pending_message(output_interface.action)

        # Store robot message to conversation history only if there was ASR input
        if "INPUT: Voice" in self.io_provider.llm_prompt:
            self.conversation_provider.store_robot_message(output_interface.action)

        state = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=self.audio_status.status_mic,
            status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
            sentence_to_speak=String(json.dumps(pending_message)),
        )

        if self.pub:
            self.pub.put(state.serialize())
            return

        self.tts.register_tts_state_callback(self.asr.audio_stream.on_tts_state_change)
        self.tts.add_pending_message(pending_message)
