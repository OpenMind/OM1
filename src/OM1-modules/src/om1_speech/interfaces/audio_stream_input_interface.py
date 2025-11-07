import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioStreamInputInterface(ABC):
    """
    Interface defining the contract for audio stream input implementations.
    All audio stream input classes must implement these methods.
    """

    @abstractmethod
    def handle_ws_incoming_message(self, connection_id: str, message: Any) -> None:
        """
        Process incoming WebSocket messages containing audio data.

        Parameters
        ----------
        connection_id : str
            Identifier for the WebSocket connection
        message : Any
            The message received from the WebSocket connection
        """
        pass

    @abstractmethod
    def setup_audio_stream(self) -> "AudioStreamInputInterface":
        """
        Set up the audio stream for processing.

        Returns
        -------
        AudioStreamInputInterface
            The current instance for method chaining
        """
        pass

    @abstractmethod
    def get_audio_chunk(self) -> Optional[bytes]:
        """
        Get the next chunk of audio data.

        Returns
        -------
        Optional[bytes]
            The next chunk of audio data, or None if no data is available
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the audio stream processing.
        """
        pass
