import logging
import random
from queue import Queue
from typing import Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector, MoveCommand
from actions.move_go2_action.interface import ActionInput
from providers.unitree_go2_odom_provider import UnitreeGo2OdomProvider
from providers.unitree_go2_rplidar_provider import UnitreeGo2RPLidarProvider
from providers.unitree_go2_state_provider import UnitreeGo2StateProvider
from unitree.unitree_sdk2py.go2.sport.sport_client import SportClient


class ActionUnitreeSDKConfig(ActionConfig):
    """
    Configuration for ActionUnitreeSDK connector.

    Parameters
    ----------
    unitree_ethernet : Optional[str]
        Ethernet channel for Unitree Go2 odometry.
    """

    unitree_ethernet: Optional[str] = Field(
        default=None,
        description="Ethernet channel for Unitree Go2 odometry.",
    )


class ActionUnitreeSDKConnector(ActionConnector[ActionUnitreeSDKConfig, ActionInput]):
    """
    This connector allows you to do the actions supported by the Unitree Go2 SDK.
    """

    def __init__(self, config: ActionUnitreeSDKConfig):
        """
        Initialize the ActionUnitreeSDK connector.

        Parameters
        ----------
        config : ActionUnitreeSDKConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        self.dog_attitude = None

        # Movement parameters
        self.turn_speed = 0.8
        self.angle_tolerance = 5.0  # degrees
        self.distance_tolerance = 0.05  # meters
        self.pending_movements: Queue[Optional[MoveCommand]] = Queue()
        self.movement_attempts = 0
        self.movement_attempt_limit = 15
        self.gap_previous = 0

        self.lidar = UnitreeGo2RPLidarProvider()
        self.unitree_go2_state = UnitreeGo2StateProvider()

        # create sport client
        self.sport_client = None
        try:
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
            self.sport_client.StopMove()
            self.sport_client.Move(0.05, 0, 0)
            logging.info("Autonomy Unitree sport client initialized")
        except Exception as e:
            logging.error(f"Error initializing Unitree sport client: {e}")

        unitree_ethernet = self.config.unitree_ethernet
        self.odom = UnitreeGo2OdomProvider(channel=unitree_ethernet)
        logging.info(f"Autonomy Odom Provider: {self.odom}")

    async def connect(self, output_interface: ActionInput) -> None:
        """
        Connect the input protocol to the Unitree Go2 action.

        Parameters
        ----------
        output_interface : ActionInput
            The input protocol containing the action details.
        """
        action = output_interface.action
        logging.info(f"ActionUnitreeSDKConnector received action: {action}")

        if action == "stand still" or action == "do nothing":
            logging.info("ActionUnitreeSDKConnector: Standing still")
            pass
        elif action == "shake paw":
            if self.unitree_go2_state.go2_action_progress == 0:
                logging.info("ActionUnitreeSDKConnector: Shaking paw")
                try:
                    if self.sport_client is not None:
                        self.sport_client.Hello()
                except Exception as e:
                    logging.error(f"Error sending ShakeHand command: {e}")
            else:
                logging.info(
                    "ActionUnitreeSDKConnector: Still performing previous action"
                )
        elif action == "dance":
            if self.unitree_go2_state.go2_action_progress == 0:
                logging.info("ActionUnitreeSDKConnector: Dancing")
                try:
                    if self.sport_client is not None:
                        dance_move = random.choice(
                            [self.sport_client.Dance1, self.sport_client.Dance2]
                        )
                        dance_move()
                except Exception as e:
                    logging.error(f"Error sending Dance command: {e}")
            else:
                logging.info(
                    "ActionUnitreeSDKConnector: Still performing previous action"
                )
        elif action == "stretch":
            if self.unitree_go2_state.go2_action_progress == 0:
                logging.info("ActionUnitreeSDKConnector: Stretching")
                try:
                    if self.sport_client is not None:
                        self.sport_client.Stretch()
                except Exception as e:
                    logging.error(f"Error sending Stretch command: {e}")
            else:
                logging.info(
                    "ActionUnitreeSDKConnector: Still performing previous action"
                )
        else:
            logging.warning(f"Action '{action}' not recognized or not implemented.")
