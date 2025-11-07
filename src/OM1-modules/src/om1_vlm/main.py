import argparse
import logging
import threading
import time
from typing import Any, Optional

from om1_utils import ws

from .config import ConfigManager
from .processor import ConnectionProcessor
from .video import VideoStream

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class Application:
    """
    Main application class for managing VLM processing and video streaming.

    Coordinates video input/output, WebSocket communication, and VLM processing
    in both server and client modes. Supports direct video device input and
    remote video streaming configurations.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration parameters:
        - ws_port : int
            WebSocket server port (default: 6789)
        - rtp_url : str
            RTP URL for video streaming (default: "rtp://192.168.1.170:1234")
        - model : str
            VLM model path/identifier (default: "Efficient-Large-Model/VILA1.5-3b")
        - max_context_len : int
            Maximum context length for VLM (default: 256)
        - max_new_tokens : int
            Maximum new tokens for generation (default: 32)
        - log_level : str
            Logging level (default: "INFO")
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ws_server: ws.Server = None

        self.config_manager = ConfigManager()

        self.connection_processor: Optional[ConnectionProcessor] = None
        self.video_device_processor: Optional[
            self.config_manager.video_device_input
        ] = None
        self.vlm_processor: Optional[self.config_manager.vlm_processor] = None

        self.video_stream: Optional[VideoStream] = None
        self.video_source: Optional[Any] = None
        self.video_output: Optional[Any] = None
        self.running: bool = False

        self.args.ws_port = self.args.ws_port or 6789
        self.args.rtp_url = self.args.rtp_url or "rtp://192.168.1.170:1234"

        # Set logging level in the lower level modules
        logging.basicConfig(level=getattr(logging, self.args.log_level.upper()))
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(
                getattr(logging, self.args.log_level.upper())
            )

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """
        Parse command-line arguments for application configuration.

        Returns
        -------
        argparse.Namespace
            Parsed command-line arguments
        """
        config_manager = ConfigManager()
        parser: argparse.ArgumentParser = config_manager.parse_model_arguments()

        # Add other agruments for WebSocket server and video streaming
        parser.add_argument(
            "--ws-host", type=str, default="localhost", help="WebSocket server host"
        )
        parser.add_argument("--ws-port", type=int, help="WebSocket server port")
        parser.add_argument(
            "--rtp-url", type=str, help="RTP URL for compressed video stream"
        )
        parser.add_argument(
            "--server-mode",
            default=False,
            action="store_true",
            help="Run in server mode",
        )
        parser.add_argument(
            "--remote-url",
            type=str,
            help="Remote webSocket URL server for video stream input",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level",
        )

        return parser.parse_args()

    # Demo for video streaming and retrieving VLM results
    # VLM results will be directly sent back to the client
    def setup_video_streaming(self):
        """
        Configure video streaming to remote WebSocket server.

        Sets up WebSocket client and video stream for sending
        video frames to a remote server.
        """
        self.ws_client = ws.Client(url=self.args.remote_url)
        self.ws_client.start()

        self.video_stream = VideoStream(self.ws_client.send_message, fps=20)
        self.video_stream.start()

    def setup_vlm_processing(self):
        """
        Configure VLM processing components.

        Sets up either server mode with connection processor or
        client mode with direct video device processing.
        """
        self.ws_server = ws.Server(host=self.args.ws_host, port=self.args.ws_port)

        if self.args.server_mode:
            self.connection_processor = ConnectionProcessor(
                self.args,
                self.config_manager.vlm_processor,
                self.config_manager.video_stream_input,
            )
            self.connection_processor.set_server(self.ws_server)
        else:
            self.vlm_processor = self.config_manager.vlm_processor(
                self.args, self.ws_server.handle_global_response
            )
            self.video_device_processor = self.config_manager.video_stream_input()
            self.video_source, self.video_output = (
                self.video_device_processor.setup_video_devices(
                    self.args, self.vlm_processor.on_video
                )
            )

            vlm_thread = threading.Thread(
                target=self.vlm_processor.process_frames,
                args=(self.video_output, self.video_source),
                daemon=True,
            )
            vlm_thread.start()

        self.ws_server.start()

    def start(self):
        """
        Start the application.

        Initializes and starts all components based on configuration.
        Handles main application loop and graceful shutdown.
        """
        logger.info("Starting application...")

        if self.args.remote_url:
            logger.info("Streaming video to WebSocket server")
            self.setup_video_streaming()
        else:
            logger.info(f"Starting VLM processing with model: {self.args.model_name}")
            self.setup_vlm_processing()

        self.running = True

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """
        Stop the application and cleanup resources.

        Performs graceful shutdown of all components in the correct order.
        """
        logger.info("Shutting down gracefully... (This may take a few seconds)")
        self.running = False

        if self.ws_server:
            self.ws_server.stop()

        if self.connection_processor:
            self.connection_processor.stop()

        if self.vlm_processor:
            self.vlm_processor.stop()

        if self.video_device_processor:
            self.video_device_processor.stop()

        if self.video_stream:
            self.video_stream.stop()

        if self.video_source:
            self.video_source.stop()

        if self.video_output:
            self.video_output.stop()


def main():
    """
    Main entry point for the application.

    Creates and runs an application instance with parsed arguments.
    """
    args = Application.parse_arguments()
    app = Application(args)
    app.start()


if __name__ == "__main__":
    main()
