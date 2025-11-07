import http.server
import logging
import socketserver
import threading

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
    """Handler for health check requests."""

    def do_GET(self):
        """Handle GET requests by returning a 200 OK status."""
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        """Override to use our logger instead of printing to stderr."""
        logger.debug(f"Health check: {format % args}")


class HealthCheckServer:
    """
    A simple HTTP server for handling AWS health checks.

    This server runs in a separate thread and responds to GET requests
    with a 200 OK status, suitable for AWS health checks.

    Parameters
    ----------
    port : int, optional
        The port to listen on, by default 8888
    """

    def __init__(self, port: int = 8888):
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        """Start the health check server in a separate thread."""
        if self.running:
            logger.warning("Health check server is already running")
            return

        def run_server():
            try:
                with socketserver.TCPServer(
                    ("", self.port), HealthCheckHandler
                ) as httpd:
                    self.server = httpd
                    logger.info(f"Health check server started on port {self.port}")
                    httpd.serve_forever()
            except Exception as e:
                logger.error(f"Health check server error: {e}")
                self.running = False

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True

    def stop(self):
        """Stop the health check server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            logger.info("Health check server stopped")

    def is_running(self) -> bool:
        """
        Check if the server is currently running.

        Returns
        -------
        bool
            True if the server is running, False otherwise
        """
        return self.running
