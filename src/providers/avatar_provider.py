import logging
from functools import wraps
from typing import Awaitable, Callable, TypeVar

from om1_utils import ws

from .singleton import singleton

R = TypeVar("R")


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


class AvatarManager:
    """
    Avatar Manager for handling avatar-related functionalities.
    """

    def __init__(self):
        pass

    @staticmethod
    def think_animation():
        """
        Decorator to send "Think" command before and "IDLE" command after function execution.

        Returns:
        --------
        function
            The decorated function.
        """

        def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
            @wraps(func)
            async def wrapper(self_instance, *args, **kwargs) -> R:
                avatar_provider = getattr(self_instance, "avatar_provider", None)

                if avatar_provider is None:
                    logging.warning("No avatar_provider found on instance")
                    return await func(self_instance, *args, **kwargs)

                prompt = args[0] if args else kwargs.get("prompt")
                if prompt and "INPUT: Voice" in prompt:
                    avatar_provider.send_avatar_command("Think")

                try:
                    result = await func(self_instance, *args, **kwargs)
                    return result
                finally:
                    avatar_provider.send_avatar_command("IDLE")

            return wrapper

        return decorator
