import logging
from typing import Optional

try:
    from unitree.unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
except ImportError:
    logging.error(
        "Unitree SDK or CycloneDDS not found. Please install the unitree_sdk2py package or CycloneDDS."
    )

from .singleton import singleton

state_machine_codes = {
    100: "Agile",
    1001: "Damping",
    1002: "Standing Lock",
    1004: "Crouch",  # Also maps to 2006
    1006: "Greeting/Stretching/Dancing/Bowing/Heart Shape/Happy",
    1007: "Sit",
    1008: "Front Jump",
    1009: "Lunge",
    1013: "Balance Standing",
    1015: "Regular Walking",
    1016: "Regular Running",
    1017: "Regular Endurance",
    1091: "Strike a Pose",
    2006: "Crouch",  # Duplicate of 1004
    2007: "Dodge",
    2008: "Bound Run",
    2009: "Jump Run",
    2010: "Classic",
    2011: "Handstand",
    2012: "Front Flip",
    2013: "Back Flip",
    2014: "Left Flip",
    2016: "Cross Step",
    2017: "Upright",
    2019: "Towing",
}


@singleton
class UnitreeGo2StateProvider:
    """
    Unitree Go2 State Provider.
    """

    def __init__(self):
        """
        Robot and sensor configuration
        """
        logging.info("Booting Unitree Go2 Sport Mode Provider")

        self.sport_mode_state_subscriber = ChannelSubscriber(
            "rt/sportmodestate", SportModeState_
        )
        self.sport_mode_state_subscriber.Init(self.SportModeStateMessageHandler, 10)

        self.go2_sport_mode_state_msg = None
        self.go2_state = None
        self.go2_state_code = None
        self.go2_action_progress = 0

    def SportModeStateMessageHandler(self, msg: SportModeState_):
        """
        Callback for handling sport mode state messages.

        Parameters
        ----------
        msg : SportModeState_
            The message containing the sport mode state data.
        """
        self.go2_sport_mode_state_msg = msg
        self.go2_state_code = msg.error_code
        self.go2_state = self.get_state_from_code(msg.error_code)
        self.go2_action_progress = msg.progress

    def get_state_from_code(self, code: int) -> Optional[str]:
        """
        Get the state name from the state code.

        Parameters
        ----------
        code : int
            The state code.

        Returns
        -------
        str
            The state name corresponding to the code, or "unknown" if not found.
        """
        return state_machine_codes.get(code, "unkown")

    @property
    def state(self) -> Optional[str]:
        """
        Get the current state of the Unitree Go2 robot.

        Returns
        -------
        Optional[str]
            The current state of the robot, or None if not available.
        """
        return self.go2_state

    @property
    def state_code(self) -> Optional[int]:
        """
        Get the current state code of the Unitree Go2 robot.

        Returns
        -------
        Optional[int]
            The current state code of the robot, or None if not available.
        """
        return self.go2_state_code

    @property
    def action_progress(self) -> int:
        """
        Get the current action progress of the Unitree Go2 robot.

        Returns
        -------
        int
            The current action progress of the robot, or 0 if not in the action mode.
        """
        return self.go2_action_progress
