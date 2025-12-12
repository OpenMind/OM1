import logging
from typing import Optional

import zenoh

from zenoh_msgs import (
    ASRText,
    AudioStatus,
    TTSInterrupt,
    open_zenoh_session,
    prepare_header,
)

from .singleton import singleton


@singleton
class TTSInterruptProvider:
    """
    Singleton provider that manages TTS interruption based on ASR input.

    """

    def __init__(self):
        """Initialize the TTS Interrupt Provider."""
        self._enabled: bool = False
        self._is_tts_active: bool = False

        self.audio_topic = "robot/status/audio"
        self.asr_topic = "om/asr/text"
        self.interrupt_topic = "robot/tts/interrupt"

        self.session: Optional[zenoh.Session] = None
        self._interrupt_pub = None
        self._audio_status: Optional[AudioStatus] = None

        try:
            self.session = open_zenoh_session()
            self._interrupt_pub = self.session.declare_publisher(self.interrupt_topic)
            self.session.declare_subscriber(self.audio_topic, self._on_audio_status)
            self.session.declare_subscriber(self.asr_topic, self._on_asr_text)
            logging.debug("TTSInterruptProvider initialized")
        except Exception as e:
            logging.error(f"Failed to initialize TTSInterruptProvider: {e}")
            self.session = None

    def enable(self):
        """Enable interrupt monitoring."""
        self._enabled = True
        logging.debug("TTSInterruptProvider enabled")

    def disable(self):
        """Disable interrupt monitoring."""
        self._enabled = False
        logging.debug("TTSInterruptProvider disabled")

    def _on_audio_status(self, data: zenoh.Sample):
        """
        Callback for audio status messages to track TTS state.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing audio status.
        """
        try:
            self._audio_status = AudioStatus.deserialize(data.payload.to_bytes())
            self._is_tts_active = (
                self._audio_status.status_speaker
                == AudioStatus.STATUS_SPEAKER.ACTIVE.value
            )
        except Exception as e:
            logging.error(f"Error processing audio status: {e}")

    def _on_asr_text(self, data: zenoh.Sample):
        """
        Callback for ASR text messages. Triggers interrupt when TTS is active.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing ASR text.
        """
        if not self._enabled:
            return

        if not self._is_tts_active:
            return

        try:
            asr_msg = ASRText.deserialize(data.payload.to_bytes())
            text = asr_msg.text

            if text and len(text.strip()) > 0:
                logging.debug(f"TTSInterruptProvider: Interrupting TTS due to ASR: {text}")
                self._publish_interrupt()
        except Exception as e:
            logging.error(f"Error handling ASR text for interrupt: {e}")

    def _publish_interrupt(self):
        """Publish a TTS interrupt message."""
        if self._interrupt_pub is None:
            logging.warning("Cannot publish interrupt: publisher not initialized")
            return

        try:
            interrupt_msg = TTSInterrupt(header=prepare_header())
            self._interrupt_pub.put(interrupt_msg.serialize())
            logging.debug("Published TTS interrupt")
        except Exception as e:
            logging.error(f"Failed to publish TTS interrupt: {e}")

    def stop(self):
        """Stop the provider and cleanup resources."""
        self._enabled = False
        if self.session:
            self.session.close()
            logging.debug("TTSInterruptProvider stopped")
