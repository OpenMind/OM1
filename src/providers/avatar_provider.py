import logging

from om1_utils import ws

from .singleton import singleton


@singleton
class AvatarProvider:
    """
    Avatar Provider.

    This class implements a singleton pattern to manage:
        * Connection to Avatar server for sending commands

    Parameters
    ----------
    avatar_server: str = "localhost"
        The Avatar server host
    avatar_port: int = 8123
        The Avatar server port
    """

    def __init__(self, avatar_server: str = "localhost", avatar_port: int = 8123):
        """
        Robot and sensor configuration
        """

        logging.info(
            f"Avatar_Provider booting Avatar Provider at server: {avatar_server}, port: {avatar_port}"
        )

        self.avatar_server_host = avatar_server
        self.avatar_server_port = avatar_port

        self.avatar_server = None
        try:
            self.avatar_server = ws.Server(
                self.avatar_server_host, self.avatar_server_port
            )
            self.avatar_server.start()
            logging.info(f"Connected to Avatar server at {avatar_server}:{avatar_port}")
        except Exception as e:
            logging.error(f"Error: {e}")

    def send_avatar_command(self, command: str):
        """
        Send command to avatar server.

        Parameters:
        -----------
        command : str
            The command string to send to the avatar server.
        """
        if self.avatar_server and self.avatar_server.running:
            self.avatar_server.handle_global_response(command)
            logging.info(f"Sent avatar command: {command}")
        else:
            logging.warning("Avatar server is not running, cannot send command.")

    def stop(self):
        """
        Stops the avatar server.
        """
        if self.avatar_server and self.avatar_server.running:
            self.avatar_server.stop()
            logging.info("Stopped Avatar Server in provider")

    @property
    def running(self) -> bool:
        """
        Check if the avatar server is running.

        Returns:
        --------
        bool
            True if the avatar server is running, False otherwise.
        """
        return self.avatar_server.running if self.avatar_server else False
