import asyncio
import json
import logging
from typing import Optional

from zenoh import ZBytes

from actions.arm_g1.interface import ArmInput
from actions.base import ActionConfig, ActionConnector
from zenoh_msgs import (
    UnitreeRequest,
    UnitreeRequestHeader,
    UnitreeRequestIdentity,
    ZenohSessionType,
    open_zenoh_session,
)

CUSTOM_API_ID = 9001
SPORT_REQUEST_TOPIC = "api/sport/request"

CUSTOM_ACTION_MAP = {
    "shake_hand": "shake_hand",
    "face_wave": "face_wave",
    "hands_up": "hands_up",
    "stand_still": "stand_still",
    "wave": "face_wave",
    "show_hand": "show_hand",
    "show_hand1": "show_hand1",
    "show_hand2": "show_hand2",
    "my_gesture": "my_gesture",
    "do_payment": "do_payment",
    "down_payment": "down_payment",
}


class ARMZenohConnector(ActionConnector[ActionConfig, ArmInput]):
    """
    Connector that sends custom arm action commands via Zenoh.

    Publishes UnitreeRequest messages to the /api/sport/request topic
    through the zenoh-bridge-ros2dds, targeting the g1_arm_action ROS2
    node (api_id=9001).
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the ARMZenohConnector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)
        self.session: Optional[ZenohSessionType] = None

        try:
            self.session = open_zenoh_session()
            logging.info("ARMZenohConnector: Zenoh session opened")
        except Exception as e:
            logging.error(f"ARMZenohConnector: Failed to open Zenoh session: {e}")

    async def connect(self, output_interface: ArmInput) -> None:
        """
        Publish the arm action command via Zenoh.

        Parameters
        ----------
        output_interface : ArmInput
            The output interface containing the arm action command.
        """
        try:
            action = output_interface.action

            if action == "idle":
                return

            if self.session is None:
                logging.error("ARMZenohConnector: No Zenoh session available")
                return

            action_name = CUSTOM_ACTION_MAP.get(action)
            if action_name is None:
                logging.warning(f"ARMZenohConnector: Unknown action '{action}'")
                return

            identity = UnitreeRequestIdentity(id=0, api_id=CUSTOM_API_ID)
            header = UnitreeRequestHeader(identity=identity)
            request = UnitreeRequest(
                header=header,
                parameter=json.dumps({"action": action_name}),
            )

            payload = ZBytes(request.serialize())
            self.session.put(SPORT_REQUEST_TOPIC, payload)
            logging.info(f"ARMZenohConnector: Published '{action}' -> action={action_name}")

            if action == "do_payment":
                asyncio.create_task(self._auto_down_payment())
        except Exception:
            logging.exception("ARMZenohConnector: Exception in connect method")

    async def _auto_down_payment(self) -> None:
        """
        Automatically issue down payment action after 10 seconds. This is triggered after do_payment is executed.
        """
        try:
            await asyncio.sleep(10)

            if self.session is None:
                logging.error("ARMZenohConnector: No Zenoh session available for auto down_payment")
                return

            action_name = CUSTOM_ACTION_MAP.get("down_payment")
            identity = UnitreeRequestIdentity(id=0, api_id=CUSTOM_API_ID)
            header = UnitreeRequestHeader(identity=identity)
            request = UnitreeRequest(
                header=header,
                parameter=json.dumps({"action": action_name}),
            )

            payload = ZBytes(request.serialize())
            self.session.put(SPORT_REQUEST_TOPIC, payload)
            logging.info("ARMZenohConnector: Auto-published 'down_payment' after 10 seconds")
        except Exception:
            logging.exception("ARMZenohConnector: Exception in auto down_payment task")

    def stop(self) -> None:
        """Close the Zenoh session."""
        if self.session:
            self.session.close()
            self.session = None
            logging.info("ARMZenohConnector: Zenoh session closed")
