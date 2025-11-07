from abc import ABC, abstractmethod

from .audio_stream_input_interface import AudioStreamInputInterface


class ASRProcessorInterface(ABC):
    """
    Interface defining the contract for ASR processor implementations.
    All ASR processor classes must implement these methods.
    """

    @abstractmethod
    def on_audio(self, audio: bytes) -> bytes:
        """
        Process incoming audio data.

        Parameters
        ----------
        audio : bytes
            Raw audio data to be processed

        Returns
        -------
        bytes
            Processed audio data
        """
        pass

    @abstractmethod
    def process_audio(self, audio_source: AudioStreamInputInterface) -> None:
        """
        Process audio stream and generate ASR transcriptions.

        Parameters
        ----------
        audio_source : AudioStreamInputInterface
            Source object that provides audio chunks for processing
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the audio processing.
        """
        pass
