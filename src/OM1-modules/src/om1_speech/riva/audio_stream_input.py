import base64
import json
import logging
from queue import Empty, Queue
from typing import Any, Dict, Optional

from ..interfaces import AudioStreamInputInterface

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioStreamInput(AudioStreamInputInterface):
    """
    A class for managing audio input streaming from WebSocket connections.

    This class provides a queue-based buffer for handling incoming audio data
    from WebSocket connections and makes it available for processing through
    a simple interface.

    Parameters
    ----------
    None
    """

    def __init__(self):
        self.running: bool = True
        self.audio_queue: Queue[Optional[Dict[str, Any]]] = Queue()

    def handle_ws_incoming_message(self, connection_id: str, message: Any):
        """
        Process incoming WebSocket messages containing audio data.

        Parameters
        ----------
        connection_id : str
            Identifier for the WebSocket connection
        message : Any
            The message received from the WebSocket connection,
            expected to be binary audio data
        """
        try:
            # Verify we received binary data
            if isinstance(message, bytes):
                logging.error("Legacy audio stream input. Set rate to 1600.")
                self.audio_queue.put(
                    {"audio": base64.b64encode(message).decode("utf-8"), "rate": 16000}
                )
            if isinstance(message, str):
                try:
                    message = json.loads(message)
                except json.JSONDecodeError:
                    logger.error("Error decoding JSON message")
                    return

                if "audio" not in message:
                    logger.error("Audio not found in message")
                    return
                audio = message["audio"]

                rate = 16000
                if "rate" in message:
                    rate = message["rate"]

                self.audio_queue.put({"audio": audio, "rate": rate})
            return
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def setup_audio_stream(self):
        """
        Set up the audio stream (placeholder method).

        Returns
        -------
        AudioStreamInput
            The current instance for method chaining
        """
        return self

    def get_audio_chunk(self) -> Optional[Dict[str, Any]]:
        try:
            data = self.audio_queue.get_nowait()
            return data
        except Empty:
            return None

    def stop(self):
        """
        Stop the audio stream processing.

        Sets the running flag to False to stop processing.
        """
        self.running = False
