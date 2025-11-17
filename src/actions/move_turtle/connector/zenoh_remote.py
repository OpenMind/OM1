import json
import logging
import time

from om1_utils import ws

from actions.base import ActionConfig, ActionConnector
from actions.move_turtle.interface import MoveInput
from providers import CommandStatus
from zenoh_msgs import geometry_msgs, open_zenoh_session


class MoveZenohRemoteConnector(ActionConnector[MoveInput]):
    """
    Zenoh remote connector for the Move action.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the Zenoh remote connector.

        Parameters
        ----------
        config : ActionConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        api_key = getattr(config, "api_key", None)
        URID = getattr(self.config, "URID", None)
        self.cmd_vel = f"{URID}/c3/cmd_vel"

        self.session = None
        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

        self.ws_client = ws.Client(
            url=f"wss://api.openmind.org/api/core/teleops/action?api_key={api_key}"
        )
        self.ws_client.start()
        self.ws_client.register_message_callback(self._on_message)

    def _on_message(self, message: str) -> None:
        """
        Callback function to handle incoming messages.

        Parameters
        ----------
        message : str
            The incoming message.
        """
        if self.session is None:
            logging.info("No open Zenoh session, returning")
            return

        try:
            command_status = CommandStatus.from_dict(json.loads(message))
            t = geometry_msgs.Twist(
                linear=geometry_msgs.Vector3(x=float(command_status.vx), y=0.0, z=0.0),
                angular=geometry_msgs.Vector3(x=0.0, y=0.0, z=float(command_status.vyaw)),
            )
            self.session.put(self.cmd_vel, t.serialize())
            logging.info(
                f"Published command: {command_status.to_dict()} - latency: {(time.time() - float(command_status.timestamp)):.3f} seconds"
            )
        except Exception as e:
            logging.error(f"Error processing message: {e}")
            return

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect to the output interface and publish the command.

        Parameters
        ----------
        output_interface : MoveInput
            The output interface for the action.
        """
        pass
