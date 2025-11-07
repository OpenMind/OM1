import argparse
import logging
import threading
import time
from typing import Optional

from om1_utils import http, ws

# Audio input from microphone
from .audio import AudioInputStream

# Multithreading
from .processor import ConnectionProcessor

# Riva ASR and TTS
from .riva import (
    ASRProcessor,
    AudioDeviceInput,
    AudioStreamInput,
    TTSProcessor,
    add_asr_config_argparse_parameters,
    add_connection_argparse_parameters,
    add_tts_argparse_parameters,
)

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class Application:
    """
    Main application class that manages speech services and server components.

    This class orchestrates the different components of the speech processing system,
    including ASR (Automatic Speech Recognition), TTS (Text-to-Speech), and audio
    streaming functionality. It can operate in either server mode for multiple
    connections or single-connection mode.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing configuration for:
        - WebSocket server (ws_host, ws_port)
        - HTTP server (http_host, http_port)
        - Server mode configuration
        - Remote URL for audio streaming
        - Logging level
        - ASR and TTS model parameters
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ws_server: Optional[ws.Server] = None
        self.http_server: Optional[http.Server] = None
        self.connection_processor: Optional[ConnectionProcessor] = None
        self.asr_processor: Optional[ASRProcessor] = None
        self.audio_source: Optional[AudioDeviceInput] = None
        self.audio_input_streamer: Optional[AudioInputStream] = None
        self.tts_processor: Optional[TTSProcessor] = None
        self.running: bool = False

        self.args.ws_port = self.args.ws_port or 6790
        self.args.http_port = self.args.http_port or 6791

        # Set logging level in the lower level modules
        logging.basicConfig(level=getattr(logging, self.args.log_level.upper()))
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(
                getattr(logging, self.args.log_level.upper())
            )

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """
        Parse command line arguments for the application.

        Returns
        -------
        argparse.Namespace
            Parsed command line arguments including:
            - Server configuration (WebSocket and HTTP)
            - Operating mode settings
            - ASR and TTS model parameters
            - Logging configuration
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--ws-host", type=str, default="localhost", help="WebSocket server host"
        )
        parser.add_argument("--ws-port", type=int, help="WebSocket server port")
        parser.add_argument(
            "--http-host", type=str, default="localhost", help="HTTP server host"
        )
        parser.add_argument("--http-port", type=int, help="HTTP server port")
        parser.add_argument(
            "--server-mode",
            default=False,
            action="store_true",
            help="Run in server mode",
        )
        parser.add_argument(
            "--remote-url",
            type=str,
            help="Remote webSocket URL server for audio stream input",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level",
        )

        # nvidia riva
        # ASR
        parser = add_asr_config_argparse_parameters(parser, profanity_filter=True)
        parser = add_connection_argparse_parameters(parser)
        # TTS
        parser = add_tts_argparse_parameters(parser)
        return parser.parse_args()

    # Demo for audio streaming and retrieving ASR results
    # ARS reulst will be directly sent back to the client
    def setup_audio_streaming(self):
        """
        Set up audio streaming to a remote WebSocket server.

        Initializes the WebSocket client and audio input streamer for
        sending audio data to a remote server.
        """
        self.ws_client = ws.Client(url=self.args.remote_url)
        self.ws_client.start()

        self.audio_input_streamer = AudioInputStream(
            audio_data_callback=self.ws_client.send_message
        )
        self.audio_input_streamer.start()

    def setup_asr_processing(self):
        """
        Set up ASR (Automatic Speech Recognition) processing.

        Configures either multi-connection server mode or single connection mode
        with default audio input. Initializes ASR processor and WebSocket server.
        """
        self.ws_server = ws.Server(host=self.args.ws_host, port=self.args.ws_port)

        # Create thread processor
        if self.args.server_mode:
            self.connection_processor = ConnectionProcessor(
                self.args, ASRProcessor, AudioStreamInput
            )
            self.connection_processor.set_server(self.ws_server)
        else:
            # Use the default audio input
            self.asr_processor = ASRProcessor(
                self.args, self.ws_server.handle_global_response
            )
            self.audio_source = AudioDeviceInput()
            self.audio_source.setup_audio_devices()

            asr_thread = threading.Thread(
                target=self.asr_processor.process_audio,
                args=(self.audio_source,),
                daemon=True,
            )
            asr_thread.start()

        self.ws_server.start()

    def setup_tts_processing(self):
        """
        Set up TTS (Text-to-Speech) processing.

        Initializes the TTS processor and HTTP server for handling
        text-to-speech requests.
        """
        self.tts_processor = TTSProcessor(self.args)

        self.http_server = http.Server(
            host=self.args.http_host, port=self.args.http_port
        )
        self.http_server.register_message_callback(self.tts_processor.process_tts)
        self.http_server.start()

        # I assume that tts service is not required to be run in a separate thread
        # tts_thread = threading.Thread(
        #     target=self.tts_processor.process_tts,
        #     daemon=True
        # )
        # tts_thread.start()

    def start(self):
        """
        Start the application and its components.

        Initializes and starts all necessary components based on the
        configuration. Runs until interrupted or stop() is called.
        """
        logger.info("Starting application...")

        if self.args.remote_url:
            logger.info("Streaming audio to WebSocket server")
            self.setup_audio_streaming()
        else:
            # hardcode the model to Riva
            logger.info("Starting ARS processing with model: Riva")
            self.setup_asr_processing()
            logger.info("Started TTS processing with model: Riva")
            self.setup_tts_processing()

        self.running = True

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """
        Stop the application and clean up resources.

        Gracefully shuts down all components and releases resources.
        This includes stopping servers, processors, and audio streams.
        """
        logger.info("Shutting down gracefully... (This may take a few seconds)")
        self.running = False

        if self.ws_server:
            self.ws_server.stop()

        if self.http_server:
            self.http_server.stop()

        if self.connection_processor:
            self.connection_processor.stop()

        if self.asr_processor:
            self.asr_processor.stop()

        if self.audio_source:
            self.audio_source.stop()

        if self.audio_input_streamer:
            self.audio_input_streamer.stop()

        if self.tts_processor:
            self.tts_processor.stop()


def main():
    """
    Main entry point for the application.
    """
    args = Application.parse_arguments()
    app = Application(args)
    app.start()


if __name__ == "__main__":
    main()
