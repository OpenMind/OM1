import json
import logging
from typing import Optional

import zenoh
from zenoh import ZBytes

from actions.arm_g1.interface import ArmInput
from actions.base import ActionConfig, ActionConnector
from zenoh_msgs import (
    UnitreeRequest,
    UnitreeRequestHeader,
    UnitreeRequestIdentity,
    open_zenoh_session,
)

# Built-in Unitree firmware action IDs (api_id=7106)
BUILTIN_ACTION_MAP = {
    "left kiss": 12,
    "right kiss": 13,
    "clap": 17,
    "high five": 18,
    "shake hand": 27,
    "heart": 20,
    "high wave": 26,
}

# Custom actions handled by g1_arm_action node (api_id=9001)
# Custom actions take priority over built-in when both exist
CUSTOM_ACTION_MAP = {
    "shake hand": "shake_hand",
    "face wave": "face_wave",
    "hands up": "hands_up",
    "stand still": "stand_still",
    "show hand": "show_hand",
    "wave": "wave",
    "move": "move",
    "show hand1": "show_hand1",
    "show hand2": "show_hand2",
    "my gesture": "my_gesture",
}

BUILTIN_API_ID = 7106
CUSTOM_API_ID = 9001
SPORT_REQUEST_TOPIC = "api/sport/request"


class ARMZenohConnector(ActionConnector[ActionConfig, ArmInput]):
    """
    Connector that sends arm action commands via Zenoh to the ROS2 /api/sport/request topic.
    Supports both built-in Unitree actions (api_id=7106) and custom arm actions (api_id=9001).
    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.session: Optional[zenoh.Session] = None

        try:
            self.session = open_zenoh_session()
            logging.info("ARMZenohConnector: Zenoh session opened")
        except Exception as e:
            logging.error(f"ARMZenohConnector: Failed to open Zenoh session: {e}")

    async def connect(self, output_interface: ArmInput) -> None:
        action = output_interface.action

        if action == "idle":
            logging.info("ARMZenohConnector: idle, no action to perform")
            return

        if self.session is None:
            logging.error("ARMZenohConnector: No Zenoh session available")
            return

        if action in CUSTOM_ACTION_MAP:
            action_name = CUSTOM_ACTION_MAP[action]
            self._publish_request(
                api_id=CUSTOM_API_ID,
                parameter=json.dumps({"action": action_name}),
            )
            logging.info(
                f"ARMZenohConnector: Custom action '{action}' -> api_id={CUSTOM_API_ID}, action={action_name}"
            )
        elif action in BUILTIN_ACTION_MAP:
            action_id = BUILTIN_ACTION_MAP[action]
            self._publish_request(
                api_id=BUILTIN_API_ID,
                parameter=json.dumps({"data": action_id}),
            )
            logging.info(
                f"ARMZenohConnector: Built-in action '{action}' -> api_id={BUILTIN_API_ID}, data={action_id}"
            )
        else:
            logging.warning(f"ARMZenohConnector: Unknown action '{action}'")

    def _publish_request(self, api_id: int, parameter: str) -> None:
        identity = UnitreeRequestIdentity(id=0, api_id=api_id)
        header = UnitreeRequestHeader(identity=identity)
        request = UnitreeRequest(header=header, parameter=parameter)

        payload = ZBytes(request.serialize())
        self.session.put(SPORT_REQUEST_TOPIC, payload)
        logging.info(
            f"ARMZenohConnector: Published to {SPORT_REQUEST_TOPIC} with api_id={api_id}"
        )

    def stop(self) -> None:
        if self.session:
            self.session.close()
            self.session = None
            logging.info("ARMZenohConnector: Zenoh session closed")
