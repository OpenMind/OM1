# The connection processor isused to handle the vlm stream and processor

import argparse
import logging
import threading
from typing import Dict, Optional

from om1_utils import ws

from ..interfaces import VideoStreamInputInterface, VLMProcessorInterface

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class ConnectionProcessor:
    """
    Manages WebSocket connections and associated VLM processing streams.

    Handles the lifecycle of video language model (VLM) processors and video streams
    for each WebSocket connection. Coordinates the creation, management, and cleanup
    of processing threads for real-time video analysis.

    Parameters
    ----------
    arg : argparse.Namespace
        Command line arguments for configuring processors and streams
    vlm_processor_class : type
        Class type for creating VLM processor instances
    video_stream_input_class : type
        Class type for creating video stream input instances
    """

    def __init__(
        self,
        arg: argparse.Namespace,
        vlm_processor_class: type,
        video_stream_input_class: type,
    ):
        self.args = arg
        self.vlm_processor_class = vlm_processor_class
        self.video_stream_input_class = video_stream_input_class
        self.vlm_processors: Dict[str, VLMProcessorInterface] = {}
        self.video_sources: Dict[str, VideoStreamInputInterface] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.ws_server: Optional[ws.Server] = None

    def set_server(self, ws_server: ws.Server):
        """
        Set up the WebSocket server and register connection callbacks.

        Parameters
        ----------
        ws_server : ws.Server
            WebSocket server instance to handle connections
        """
        self.ws_server = ws_server

        self.ws_server.register_connection_callback(
            lambda event, conn_id: self.handle_connection_event(event, conn_id)
        )

    def handle_connection_event(self, event: str, connection_id: str):
        """
        Handle WebSocket connection events.

        Parameters
        ----------
        event : str
            Type of connection event ('connect' or 'disconnect')
        connection_id : str
            Unique identifier for the connection
        """
        if event == "connect":
            self.handle_new_connection(connection_id)
        elif event == "disconnect":
            self.handle_connection_closed(connection_id)

    def handle_new_connection(self, connection_id: str):
        """
        Set up processing components for a new connection.

        Creates and initializes:
        - VLM processor
        - Video stream source
        - Processing thread

        Parameters
        ----------
        connection_id : str
            Unique identifier for the new connection
        """
        vlm_processor = self.vlm_processor_class(
            self.args,
            callback=lambda message: self.ws_server.handle_response(
                connection_id, message
            ),
        )

        self.vlm_processors[connection_id] = vlm_processor

        # Create video stream source for this connection
        video_source = self.video_stream_input_class()
        video_source.setup_video_stream(self.args, vlm_processor.on_video)
        self.video_sources[connection_id] = video_source

        # Register message callback for video stream
        self.ws_server.register_message_callback(
            connection_id,
            lambda conn_id, message: video_source.handle_ws_incoming_message(
                conn_id, message
            ),
        )

        processing_thread = threading.Thread(
            target=vlm_processor.process_frames,
            args=(
                None,
                video_source,
            ),
        )
        self.processing_threads[connection_id] = processing_thread
        processing_thread.start()

        logger.info(f"Started processing thread for connection {connection_id}")

    def handle_connection_closed(self, connection_id: str):
        """
        Clean up resources when a connection closes.

        Parameters
        ----------
        connection_id : str
            Unique identifier for the closed connection
        """
        if connection_id in self.vlm_processors:
            self.vlm_processors[connection_id].stop()
            del self.vlm_processors[connection_id]

        if connection_id in self.video_sources:
            self.video_sources[connection_id].stop()
            del self.video_sources[connection_id]

        if connection_id in self.processing_threads:
            del self.processing_threads[connection_id]

        logger.info(f"Stopped processing thread for connection {connection_id}")

    def stop(self):
        """
        Stop all active connections and cleanup resources.
        """
        for connection_id in list(self.vlm_processors.keys()):
            self.handle_connection_closed(connection_id)
