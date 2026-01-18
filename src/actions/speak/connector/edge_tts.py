import asyncio
import logging
import tempfile
from pathlib import Path

import edge_tts
import zenoh
from pydantic import Field
from pydub import AudioSegment
from pydub.playback import play

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput
from zenoh_msgs import (
    String,
    TTSStatusRequest,
    TTSStatusResponse,
    open_zenoh_session,
    prepare_header,
)


class SpeakEdgeTTSConfig(ActionConfig):
    """
    Configuration for Edge TTS connector.

    Parameters
    ----------
    voice : str
        Edge TTS voice name (default is "en-US-AriaNeural").
    rate : str
        Speech rate adjustment (default is "+0%").
    volume : str
        Volume adjustment (default is "+0%").
    """

    voice: str = Field(
        default="en-US-AriaNeural",
        description="Edge TTS voice name",
    )
    rate: str = Field(
        default="+0%",
        description="Speech rate adjustment",
    )
    volume: str = Field(
        default="+0%",
        description="Volume adjustment",
    )


class SpeakEdgeTTSConnector(ActionConnector[SpeakEdgeTTSConfig, SpeakInput]):
    """
    A "Speak" connector that uses Microsoft Edge TTS for text-to-speech.

    This connector provides free, high-quality TTS without requiring API keys.
    It is compatible with the standard SpeakInput interface.
    """

    def __init__(self, config: SpeakEdgeTTSConfig):
        """
        Initialize the connector.

        Parameters
        ----------
        config : SpeakEdgeTTSConfig
            Configuration for the connector.
        """
        super().__init__(config)

        self.voice = self.config.voice
        self.rate = self.config.rate
        self.volume = self.config.volume

        # Zenoh topics for TTS control
        self.tts_status_request_topic = "om/tts/request"
        self.tts_status_response_topic = "om/tts/response"

        self.session = None
        self.tts_enabled = True

        # Initialize Zenoh session for TTS control
        try:
            self.session = open_zenoh_session()
            self.session.declare_subscriber(
                self.tts_status_request_topic, self._zenoh_tts_status_request
            )
            self._zenoh_tts_status_response_pub = self.session.declare_publisher(
                self.tts_status_response_topic
            )
            logging.info("Edge TTS Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Edge TTS Zenoh client: {e}")

        logging.info(
            f"Edge TTS connector initialized with voice={self.voice}, rate={self.rate}, volume={self.volume}"
        )

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        Process a speak action by generating and playing audio with Edge TTS.

        Parameters
        ----------
        output_interface : SpeakInput
            The SpeakInput interface containing the text to be spoken.
        """
        if not self.tts_enabled:
            logging.info("TTS is disabled, skipping speak action")
            return

        text = output_interface.action
        logging.info(f"Edge TTS: {text}")

        try:
            await self._generate_and_play(text)
        except Exception as e:
            logging.error(f"Error in Edge TTS: {e}")

    async def _generate_and_play(self, text: str):
        """
        Generate audio from text and play it.

        Parameters
        ----------
        text : str
            Text to convert to speech
        """
        temp_path = None
        try:
            # Create a temporary file for the audio
            # delete=False is required to let other processes/libs read it before deletion
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            # Generate audio using edge-tts
            communicate = edge_tts.Communicate(
                text=text, voice=self.voice, rate=self.rate, volume=self.volume
            )

            await communicate.save(str(temp_path))

            logging.debug(f"Edge TTS: Audio generated and saved to {temp_path}")

            # Load and play audio using pydub
            audio = AudioSegment.from_mp3(str(temp_path))
            logging.info(f"Edge TTS: Playing audio ({len(audio)}ms)")

            # FIX: Use asyncio.to_thread to prevent blocking the event loop
            await asyncio.to_thread(play, audio)

        except Exception as e:
            logging.error(f"Error generating/playing audio with Edge TTS: {e}")
            # We don't re-raise here to avoid crashing the whole connector loop,
            # just log error as per connect() logic.
            # If you want to bubble up, add 'raise' here.

        finally:
            # FIX: Ensure cleanup happens even if playback fails
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logging.debug("Edge TTS: Temp file cleaned up")
                except Exception as cleanup_err:
                    logging.warning(
                        f"Edge TTS: Failed to delete temp file: {cleanup_err}"
                    )

    def _zenoh_tts_status_request(self, data: zenoh.Sample):
        """
        Process an incoming TTS control status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received.
        """
        try:
            tts_status = TTSStatusRequest.deserialize(data.payload.to_bytes())
            logging.debug(f"Received TTS Control Status message: {tts_status}")

            code = tts_status.code
            request_id = tts_status.request_id

            # Read the current status (code == 2)
            if code == 2:
                response_code = 1 if self.tts_enabled else 0
                status_text = "TTS Enabled" if self.tts_enabled else "TTS Disabled"

            # Enable the TTS (code == 1)
            elif code == 1:
                self.tts_enabled = True
                logging.info("Edge TTS Enabled")
                response_code = 1
                status_text = "Edge TTS Enabled"

            # Disable the TTS (code == 0)
            elif code == 0:
                self.tts_enabled = False
                logging.info("Edge TTS Disabled")
                response_code = 0
                status_text = "Edge TTS Disabled"

            else:
                return  # Unknown code

            tts_status_response = TTSStatusResponse(
                header=prepare_header(tts_status.header.frame_id),
                request_id=request_id,
                code=response_code,
                status=String(data=status_text),
            )

            if self._zenoh_tts_status_response_pub:
                self._zenoh_tts_status_response_pub.put(tts_status_response.serialize())

        except Exception as e:
            logging.error(f"Error processing Zenoh TTS status request: {e}")

    def stop(self) -> None:
        """
        Stop the Edge TTS connector and cleanup resources.
        """
        if self.session:
            self.session.close()
            logging.info("Edge TTS Zenoh client closed")
