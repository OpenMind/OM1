import logging

from backgrounds.base import Background, BackgroundConfig
from providers.avatar_provider import AvatarProvider


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

        self.avatar_provider = AvatarProvider(
            avatar_server=self.avatar_server_host,
            avatar_port=self.avatar_server_port,
        )
        logging.info("Initiated Avatar Provider in background")

    def stop(self):
        self.avatar_provider.stop()
        logging.info("Stopped Avatar Provider in background")
