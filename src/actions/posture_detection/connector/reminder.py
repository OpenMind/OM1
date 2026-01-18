import json
import logging
import time
from typing import Dict, Optional
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.posture_detection.interface import (
    PostureDetectionInput,
    PostureSeverity,
    PostureType,
)
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.health_detection_provider import HealthDetectionProvider
from providers.io_provider import IOProvider
from providers.teleops_conversation_provider import TeleopsConversationProvider
from zenoh_msgs import (
    AudioStatus,
    String,
    TTSStatusRequest,
    open_zenoh_session,
    prepare_header,
)


class PostureReminderConfig(ActionConfig):
    """
    Configuration for Posture Detection Reminder connector.

    Parameters
    ----------
    elevenlabs_api_key : Optional[str]
        ElevenLabs API key for TTS.
    voice_id : str
        ElevenLabs voice ID.
    model_id : str
        ElevenLabs model ID.
    reminder_interval_minutes : float
        Minimum time between reminders for the same person (in minutes).
    enable_gentle_reminders : bool
        Whether to use gentle, encouraging reminders instead of warnings.
    """

    elevenlabs_api_key: Optional[str] = Field(
        default=None, description="ElevenLabs API key"
    )
    voice_id: str = Field(
        default="JBFqnCBsd6RMkjVDRZzb",
        description="ElevenLabs voice ID",
    )
    model_id: str = Field(
        default="eleven_flash_v2_5",
        description="ElevenLabs model ID",
    )
    reminder_interval_minutes: float = Field(
        default=30.0, description="Minimum time between reminders (minutes)"
    )
    enable_gentle_reminders: bool = Field(
        default=True, description="Use gentle, encouraging reminders"
    )


class PostureReminderConnector(
    ActionConnector[PostureReminderConfig, PostureDetectionInput]
):
    """
    Connector that handles posture detection and provides gentle reminders.
    """

    def __init__(self, config: PostureReminderConfig):
        """
        Initialize the PostureReminderConnector.

        Parameters
        ----------
        config : PostureReminderConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        self.io_provider = IOProvider()
        self.health_provider = HealthDetectionProvider()
        self.last_reminder_times: Dict[str, float] = {}  # person_name -> timestamp

        # Eleven Labs TTS configuration
        elevenlabs_api_key = self.config.elevenlabs_api_key
        voice_id = self.config.voice_id
        model_id = self.config.model_id

        self.audio_topic = "robot/status/audio"
        self.tts_status_request_topic = "om/tts/request"
        self.session = None
        self.audio_pub = None

        self.audio_status = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=AudioStatus.STATUS_MIC.UNKNOWN.value,
            status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
            sentence_to_speak=String(""),
        )

        try:
            self.session = open_zenoh_session()
            self.audio_pub = self.session.declare_publisher(self.audio_topic)
            self.session.declare_subscriber(self.audio_topic, self.zenoh_audio_message)
            self.session.declare_subscriber(
                self.tts_status_request_topic, self._zenoh_tts_status_request
            )

            if self.audio_pub:
                self.audio_pub.put(self.audio_status.serialize())

            logging.info("Posture Reminder Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Posture Reminder Zenoh client: {e}")

        self.tts = ElevenLabsTTSProvider(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format="mp3_44100_128",
        )
        self.tts.start()

        self.tts_enabled = True
        self.conversation_provider = TeleopsConversationProvider(api_key=api_key)

    def zenoh_audio_message(self, data: zenoh.Sample):
        """
        Process an incoming audio status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

    async def connect(self, output_interface: PostureDetectionInput) -> None:
        """
        Handle posture detection and provide appropriate reminders.

        Parameters
        ----------
        output_interface : PostureDetectionInput
            The input protocol containing posture detection details.
        """
        # Record the posture detection
        self.health_provider.record_posture(
            posture_type=output_interface.posture_type.value,
            severity=output_interface.severity.value,
            duration_minutes=output_interface.duration_minutes,
            person_name=output_interface.person_name,
        )

        # Check if we should send a reminder
        person_key = output_interface.person_name or "unknown"
        last_reminder = self.last_reminder_times.get(person_key, 0)

        if not self.health_provider.should_remind_posture(
            person_key, last_reminder if last_reminder > 0 else None
        ):
            logging.debug(f"Skipping reminder for {person_key}, too soon since last reminder")
            return

        # Only remind for poor posture (not good posture)
        if output_interface.posture_type == PostureType.GOOD:
            logging.debug("Good posture detected, no reminder needed")
            return

        if self.tts_enabled is False:
            logging.info("TTS is disabled, skipping posture reminder")
            return

        # Generate reminder message
        reminder_message = self._generate_reminder_message(output_interface)

        # Add pending message to TTS
        pending_message = self.tts.create_pending_message(reminder_message)

        # Store robot message to conversation history
        if (
            self.io_provider.llm_prompt is not None
            and "INPUT: Voice" in self.io_provider.llm_prompt
        ):
            self.conversation_provider.store_robot_message(reminder_message)

        # Avoid queuing too many TTS messages
        if self.tts.get_pending_message_count() > 0:
            logging.warning("Too many pending TTS messages, skipping posture reminder")
            return

        state = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=self.audio_status.status_mic,
            status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
            sentence_to_speak=String(json.dumps(pending_message)),
        )

        if self.audio_pub:
            self.audio_pub.put(state.serialize())
            return

        # Fallback: use TTS directly if Zenoh is not available
        self.tts.add_pending_message(pending_message)

        # Update last reminder time
        self.last_reminder_times[person_key] = time.time()

        logging.info(
            f"Posture reminder sent: {output_interface.posture_type.value} "
            f"for {output_interface.person_name or 'unknown person'}"
        )

    def _generate_reminder_message(
        self, input_interface: PostureDetectionInput
    ) -> str:
        """
        Generate reminder message based on posture detection input.

        Parameters
        ----------
        input_interface : PostureDetectionInput
            The posture detection input.

        Returns
        -------
        str
            The reminder message.
        """
        person = input_interface.person_name if input_interface.person_name else "You"
        duration = int(input_interface.duration_minutes)

        if self.config.enable_gentle_reminders:
            # Gentle, encouraging reminders
            messages = {
                PostureType.SLUMPED: f"{person}, I notice you've been slouching for {duration} minutes. Let's sit up straight to protect your back!",
                PostureType.HUNCHED: f"{person}, your shoulders seem rounded. Try rolling them back and lifting your chin - it'll help reduce neck strain!",
                PostureType.LEANING: f"{person}, you're leaning to one side. Let's adjust to center your posture for better balance!",
                PostureType.ASYMMETRIC: f"{person}, your posture looks a bit uneven. Try to align your shoulders and keep your spine straight!",
                PostureType.LAYING: f"{person}, you've been laying down for a while. Consider taking a short walk to refresh yourself!",
            }
        else:
            # More direct reminders
            messages = {
                PostureType.SLUMPED: f"Posture alert: {person} has been slouching for {duration} minutes. Please sit up straight.",
                PostureType.HUNCHED: f"Posture alert: {person} has rounded shoulders. Please adjust your posture.",
                PostureType.LEANING: f"Posture alert: {person} is leaning to one side. Please center your posture.",
                PostureType.ASYMMETRIC: f"Posture alert: {person} has uneven posture. Please align your body.",
                PostureType.LAYING: f"Notice: {person} has been laying down for {duration} minutes. Consider getting up.",
            }

        base_message = messages.get(input_interface.posture_type, "Please adjust your posture.")

        # Add recommendation if provided
        if input_interface.recommendation:
            base_message += f" {input_interface.recommendation}"

        # Adjust tone based on severity
        if input_interface.severity == PostureSeverity.SEVERE:
            base_message = f"Important: {base_message} This is important for your long-term health."

        return base_message

    def _zenoh_tts_status_request(self, data: zenoh.Sample):
        """
        Process an incoming TTS control status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received.
        """
        tts_status = TTSStatusRequest.deserialize(data.payload.to_bytes())
        logging.debug(f"Received TTS Control Status message: {tts_status}")

        code = tts_status.code

        if code == 1:
            self.tts_enabled = True
            logging.debug("TTS Enabled")
        elif code == 0:
            self.tts_enabled = False
            logging.debug("TTS Disabled")

