import logging

from om1_utils import ws

from backgrounds.base import Background, BackgroundConfig

from providers.singleton import singleton

@singleton
class Avatar(Background):
    """
    Manages connection to Avatar server for sending commands.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        self.avatar_server_host = getattr(self.config, "avatar_server", "localhost")
        logging.info(f"Avatar using server host: {self.avatar_server_host}")

        self.avatar_server_port = getattr(self.config, "avatar_port", 8123)
        logging.info(f"Avatar using server port: {self.avatar_server_port}")

        self.avatar_server = ws.Server(self.avatar_server_host, self.avatar_server_port)
        self.avatar_server.start()
        logging.info("Initiated Avatar Server in background")


    def send_avatar_command(self, command: str):
        """
        Send command to avatar server.

        Parameters:
        -----------
        command : str
            The command string to send to the avatar server.
        """
        if self.avatar_server.running:
            self.avatar_server.handle_global_response(command)
            logging.info(f"Sent avatar command: {command}")
        else:
            logging.warning("Avatar server is not running, cannot send command.")

    def stop(self):
        """
        Stops the avatar server.
        """
        if self.avatar_server.running:
            self.avatar_server.stop()
            logging.info("Stopped Avatar Server in background")
