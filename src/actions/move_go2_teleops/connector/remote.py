import json
import logging
import time
from enum import Enum

from om1_utils import ws
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.move_go2_teleops.interface import MoveInput
from providers import CommandStatus
from providers.unitree_go2_state_provider import UnitreeGo2StateProvider
from unitree.unitree_sdk2py.go2.sport.sport_client import SportClient


class RobotState(Enum):
    """
    Enumeration of possible robot states.
    """

    STANDING = "standing"
    SITTING = "sitting"


class MoveGo2RemoteConfig(ActionConfig):
    """
    Configuration for MoveGo2Remote connector.

    Parameters
    ----------
    api_key : Optional[str]
        OM API key.
    """

    api_key: str = Field(
        default="",
        description="OM API key.",
    )


class MoveGo2RemoteConnector(ActionConnector[MoveGo2RemoteConfig, MoveInput]):
    """
    Remote connector for the Move Go2 teleops action.

    NOTE: This connector has been deprecated. OM1 Orchestrator docker automatically supports remote teleops
    """

    def __init__(self, config: MoveGo2RemoteConfig):
        """
        Initialize the MoveGo2Remote connector.

        Parameters
        ----------
        config : MoveGo2RemoteConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        api_key = self.config.api_key

        self.sport_client = None
        try:
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
            logging.info("Unitree sport client initialized")
        except Exception as e:
            logging.error(f"Error initializing Unitree sport client: {e}")

        self.ws_client = ws.Client(
            url=f"wss://api.openmind.org/api/core/teleops/action?api_key={api_key}"
        )
        self.ws_client.register_message_callback(self._on_message)
        self.ws_client.start()

        self.unitree_state_provider = UnitreeGo2StateProvider()

    def _on_message(self, message: str) -> None:
        """
        Callback function to handle incoming messages.

        Parameters
        ----------
        message : str
            The incoming message.
        """
        if self.sport_client is None:
            logging.info("No open Unitree sport client, returning")
            return

        if self.unitree_state_provider.state_code == 1002:
            self.sport_client.BalanceStand()

        if self.unitree_state_provider.action_progress != 0:
            logging.info(
                f"Action in progress: {self.unitree_state_provider.action_progress}"
            )
            return

        try:
            command_status = CommandStatus.from_dict(json.loads(message))
            self.sport_client.Move(
                command_status.vx, command_status.vy, command_status.vyaw
            )
            logging.info(
                f"Published command: {command_status.to_dict()} - latency: {(time.time() - float(command_status.timestamp)):.3f} seconds"
            )
        except Exception as e:
            logging.error(f"Error processing command status: {e}")

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect to the output interface and publish the command.

        Parameters
        ----------
        output_interface : MoveInput
            The output interface for the action.
        """
        pass
