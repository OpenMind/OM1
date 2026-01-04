import logging
import time
from typing import Optional

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput, MovementAction
from capabilities import CapabilityDescriptor, ComponentType, Constraint


class MoveUnitreeSDKConnector(ActionConnector[ActionConfig, MoveInput]):

    def __init__(self, config: ActionConfig):
        super().__init__(config)

    def get_capabilities(self) -> Optional[CapabilityDescriptor]:
        """
        Get capability descriptor for the move action.

        Returns
        -------
        CapabilityDescriptor
            Describes available movement actions and constraints
        """
        # Get all supported movement actions from the enum
        supported_movements = [action.value for action in MovementAction]

        return CapabilityDescriptor(
            component_name="move",
            component_type=ComponentType.ACTION,
            supported_features=supported_movements,
            constraints=[
                Constraint(name="max_speed", value=1.2, unit="m/s"),
                Constraint(name="max_rotation", value=45, unit="deg/s"),
            ],
            is_available=True,
            description="Movement control for Unitree robot via ROS2",
            metadata={
                "connector_type": "ros2",
                "robot_type": "unitree",
            },
        )

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect the input protocol to the move action via Unitree SDK.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol containing the action details.
        """
        new_msg = {"move": ""}

        # stub to show how to do this
        if output_interface.action == "stand still":
            new_msg["move"] = "stand still"
        elif output_interface.action == "sit":
            new_msg["move"] = "sit"
        elif output_interface.action == "dance":
            new_msg["move"] = "dance"
        elif output_interface.action == "shake paw":
            new_msg["move"] = "shake paw"
        elif output_interface.action == "walk":
            new_msg["move"] = "walk"
        elif output_interface.action == "walk back":
            new_msg["move"] = "walk back"
        elif output_interface.action == "run":
            new_msg["move"] = "run"
        elif output_interface.action == "jump":
            new_msg["move"] = "jump"
        elif output_interface.action == "wag tail":
            new_msg["move"] = "wag tail"
        else:
            logging.info(f"Other move type: {output_interface.action}")
            # raise ValueError(f"Unknown move type: {output_interface.action}")

        logging.info(f"SendThisToROS2: {new_msg}")

    def tick(self) -> None:
        time.sleep(0.1)
        # logging.info("MoveUnitreeSDKConnector Tick")
