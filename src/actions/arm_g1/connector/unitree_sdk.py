import logging

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
            logging.error(f"Failed to initialize G1 Arm Action Client: {e}")

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

        # Custom actions (api_id=9001) are not supported via the SDK connector.
        # Use the Zenoh connector for custom action support.
        custom_actions = {
            "shake hand",
            "face wave",
            "hands up",
            "stand still",
            "show hand",
            "wave",
            "move",
            "show hand1",
            "show hand2",
            "my gesture",
        }
        if output_interface.action in custom_actions:
            logging.warning(
                f"Custom action '{output_interface.action}' is not supported via SDK connector. "
                "Use the Zenoh connector for custom actions."
            )
            return

        builtin_action_map = {
            "left kiss": 12,
            "right kiss": 13,
            "clap": 17,
            "high five": 18,
            "heart": 20,
            "high wave": 26,
        }

        action_id = builtin_action_map.get(output_interface.action)
        if action_id is None:
            logging.warning(f"Unknown action: {output_interface.action}")
            return

        logging.info(f"Executing action with ID: {action_id}")
        self.client.ExecuteAction(action_id)
