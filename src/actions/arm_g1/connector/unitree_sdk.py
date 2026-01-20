import logging
from typing import Optional

from actions.arm_g1.interface import ArmInput
from actions.base import ActionConfig, ActionConnector
from unitree.unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient


class ARMUnitreeSDKConnector(ActionConnector[ActionConfig, ArmInput]):
    """
    Connector that interacts with the G1 Arm Action Client to perform arm actions.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the ARMUnitreeSDKConnector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        try:
            self.client = G1ArmActionClient()
            self.client.SetTimeout(10.0)
            self.client.Init()
            logging.info("G1 Arm Action Client initialized successfully.")
        except Exception as e:
            logging.error(
                f"Failed to initialize G1 Arm Action Client: {e}",
                exc_info=True,
            )

    async def connect(self, output_interface: ArmInput) -> None:
        """
        Connects to the G1 Arm Action Client and executes the specified action.

        Parameters
        ----------
        output_interface : ArmInput
            The output interface containing the arm action command.
        """
        logging.info(f"AI command.action: {output_interface.action}")

        if output_interface.action == "idle":
            logging.info("No action to perform, returning.")
            return

        action_id: Optional[int] = None

        if output_interface.action == "left kiss":
            action_id = 12
        elif output_interface.action == "right kiss":
            action_id = 13
        elif output_interface.action == "clap":
            action_id = 17
        elif output_interface.action == "high five":
            action_id = 18
        elif output_interface.action == "shake hand":
            action_id = 27
        elif output_interface.action == "heart":
            action_id = 20
        elif output_interface.action == "high wave":
            action_id = 26
        else:
            logging.warning(f"Unknown action: {output_interface.action}")
            return

        if action_id is None:
            logging.error(f"Action ID is None for action: {output_interface.action}")
            return

        try:
            logging.info(f"Executing action with ID: {action_id}")
            self.client.ExecuteAction(action_id)
        except Exception as e:
            logging.error(
                f"Failed to execute arm action (ID: {action_id}, action: {output_interface.action}): {e}",
                exc_info=True,
            )
            raise
