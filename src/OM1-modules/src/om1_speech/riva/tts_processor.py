import argparse
import base64
import logging
from typing import Any, Dict, Optional

# riva is only available on Jetson devices
try:
    from riva import client
    from riva.client import SpeechSynthesisService
except ModuleNotFoundError:
    client = None

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class TTSProcessor:
    """
    A class for processing text-to-speech synthesis using NVIDIA Riva.

    Parameters
    ----------
    model_args : argparse.Namespace
    model_args : argparse.Namespace
        Command line arguments and configuration parameters for the TTS model.
        Expected arguments include:
        - ssl_cert: SSL certificate path
        - use_ssl: Whether to use SSL
        - server: Server address
        - metadata: Additional metadata
        - voice: Voice identifier for synthesis
        - language_code: Language code for TTS
        - tts_sample_rate_hz: Audio sample rate in Hz
        - audio_prompt_file: Path to audio prompt file
        - quality: Audio quality setting (1-100)
    """

    def __init__(self, model_args: argparse.Namespace):
        self.model: Optional[SpeechSynthesisService] = None
        self.args = model_args
        self.running: bool = True

        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the Riva TTS model and configuration.

        Sets up the TTS service with specified authentication and performs
        a test synthesis to ensure the model is loaded properly.

        Raises
        ------
        Exception
            If there are errors during model initialization
        """
        auth = client.Auth(
            self.args.ssl_cert, self.args.use_ssl, self.args.server, self.args.metadata
        )
        self.model = client.SpeechSynthesisService(auth)

        # hot load the model
        try:
            self.generate_tts("om1 is ready")
        except Exception as e:
            logger.error(f"Error initializing TTS model: {e}")

    def generate_tts(self, text: str):
        """
        Generate speech from input text using the TTS model.

        Parameters
        ----------
        text : str
            The input text to synthesize into speech

        Returns
        -------
        object
            Riva synthesis result containing audio data and metadata

        Notes
        -----
        The synthesis is performed with the configured voice, language,
        sample rate, and quality settings specified in model_args.
        """
        return self.model.synthesize(
            text,
            self.args.voice,
            self.args.language_code,
            sample_rate_hz=self.args.tts_sample_rate_hz,
            audio_prompt_file=self.args.audio_prompt_file,
            quality=20 if self.args.quality is None else self.args.quality,
            custom_dictionary={},
        )

    def process_tts(self, tts_input: Dict[Any, Any], *args, **kwargs):
        """
        Process a TTS request and return base64 encoded audio.

        Parameters
        ----------
        tts_input : Dict[Any, Any]
            Input dictionary containing the 'text' key with the text to synthesize
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments

        Returns
        -------
        Optional[Dict[str, str]]
            Dictionary containing:
            - 'response': Base64 encoded audio data
            - 'content_type': Audio format identifier
            Returns None if processing fails
        """
        try:
            if isinstance(tts_input, dict):
                tts_input = tts_input["text"]
            else:
                return None

            logger.info(f"Processing TTS: {tts_input}")
            result = self.generate_tts(tts_input)
            audio_b64 = base64.b64encode(result.audio).decode("utf-8")

            return {"response": audio_b64, "content_type": "audio/mp3"}
        except Exception as e:
            logger.error(f"Error processing TTS: {e}")
            return None

    def stop(self):
        """
        Stop the TTS processor.

        Sets the running flag to False to stop processing.
        """
        self.running = False
