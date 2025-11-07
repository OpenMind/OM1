import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, Optional, Union

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)

# Type aliases for JSON data and request handling
JsonDict = Dict[str, Any]
JsonResponse = Union[Dict[str, Any], str]
RequestCallback = Callable[[JsonDict, str], JsonResponse]


class _RequestHandler(BaseHTTPRequestHandler):
    """
    Custom HTTP request handler for processing JSON POST requests.

    This handler only accepts POST requests and returns JSON responses. It processes
    requests through a callback function that handles the business logic.

    Parameters
    ----------
    callback : Optional[RequestCallback]
        Function to process incoming requests, takes JSON data and path as input
    *args : tuple
        Variable length argument list for BaseHTTPRequestHandler
    **kwargs : dict
        Arbitrary keyword arguments for BaseHTTPRequestHandler
    """

    def __init__(
        self, message_callback: Optional[RequestCallback] = None, *args, **kwargs
    ):
        self.message_callback = message_callback
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """
        Handle GET requests by returning a method not allowed error.
        """
        self.send_error_response(405, "GET method not allowed")

    def do_POST(self):
        """
        Handle POST requests by processing JSON data through the callback.

        Reads the POST data, parses it as JSON, and passes it to the callback
        function if one is registered. Sends the callback's response back to
        the client.
        """
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            logger.info(f"Received POST data: {post_data}")
            json_data: Dict = json.loads(post_data)

            # Process request through callback and get response
            if self.message_callback:
                response = self.message_callback(json_data, self.path)
                if response:
                    self.send_json_response(response)
                else:
                    self.send_error_response(500, "No response from callback handler")
            else:
                self.send_error_response(500, "No callback handler registered")

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self.send_error_response(400, str(e))

    def send_json_response(self, data: JsonResponse):
        """
        Send a JSON response to the client.

        Parameters
        ----------
        data : Union[Dict[str, Any], str]
            The data to send as JSON response. Can be either a dictionary
            or a string (which will be wrapped in a response object)
        """
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            if isinstance(data, str):
                response_json = json.dumps({"response": data})
            else:
                response_json = json.dumps(data)

            self.wfile.write(response_json.encode("utf-8"))
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            self.send_error_response(500, "Error formatting response")

    def send_error_response(self, code: int, message: str):
        """
        Send an error response to the client.

        Parameters
        ----------
        code : int
            HTTP status code for the error
        message : str
            Error message to include in the response
        """
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        error_response = json.dumps({"error": message})
        self.wfile.write(error_response.encode("utf-8"))


class Server:
    """
    HTTP server implementation that handles JSON POST requests.

    This server runs in a separate thread and processes incoming POST requests
    through a registered callback function.

    Parameters
    ----------
    host : str, optional
        The hostname to bind the server to, by default "localhost"
    port : int, optional
        The port number to listen on, by default 6791
    timeout : int, optional
        Server timeout in seconds, by default 15
    """

    def __init__(self, host: str = "localhost", port: int = 6791, timeout: int = 15):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.running: bool = True
        self.server: Optional[HTTPServer] = None
        self.message_callback: Optional[RequestCallback] = None

    def _create_handler(self, *args, **kwargs) -> _RequestHandler:
        """
        Create a new request handler instance.

        Parameters
        ----------
        *args : tuple
            Variable length argument list for RequestHandler
        **kwargs : dict
            Arbitrary keyword arguments for RequestHandler

        Returns
        -------
        _RequestHandler
            A new request handler instance with the registered callback
        """
        return _RequestHandler(self.message_callback, *args, **kwargs)

    def _run_server(self):
        """
        Internal method to run the server and handle requests.

        Creates and runs the HTTP server, processing requests until stopped.
        """
        self.server = HTTPServer((self.host, self.port), self._create_handler)
        self.server.timeout = self.timeout
        logger.info(f"HTTP server started on {self.host}:{self.port}")

        while self.running:
            try:
                self.server.handle_request()
            except Exception as e:
                logger.error(f"Error handling request: {e}")
                if not self.running:
                    break

    def start(self):
        """
        Start the HTTP server in a separate thread.
        """
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        logger.info("HTTP server thread started")

    def register_message_callback(self, message_callback: RequestCallback):
        """
        Register a callback function for processing requests.

        Parameters
        ----------
        callback : Callable[[Dict[str, Any], str], Union[Dict[str, Any], str]]
            Function that takes JSON data and path as input and returns
            a response
        """
        self.message_callback = message_callback
        logger.info("Registered request callback")

    def stop(self):
        """
        Stop the HTTP server.

        Stops the server and closes all connections.
        """
        self.running = False
        if self.server:
            self.server.server_close()
        logger.info("HTTP server stopped")
