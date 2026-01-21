import concurrent.futures
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.move_ub.interface import MoveInput
from ubtech.ubtechapi import YanAPI


@dataclass
class Motion:
    """
    Represents a motion command for the UBTECH Yanshee robot.
    """

    name: str
    direction: Optional[str] = field(default=None)
    speed: Optional[str] = field(default=None)
    repeat: Optional[int] = field(default=None)
    version: Optional[str] = field(default=None)

    # map name → defaults
    _defaults = {
        "reset": {"direction": "", "speed": "normal", "repeat": 1, "version": "v1"},
        "wave": {"direction": "both", "speed": "normal", "repeat": 1, "version": "v1"},
        "bow": {"direction": "", "speed": "normal", "repeat": 1, "version": "v1"},
        "crouch": {"direction": "", "speed": "normal", "repeat": 1, "version": "v1"},
        "come on": {"direction": "", "speed": "normal", "repeat": 1, "version": "v1"},
        "walk": {
            "direction": "forward",
            "speed": "normal",
            "repeat": 1,
            "version": "v1",
        },
        "head": {
            "direction": "forward",
            "speed": "normal",
            "repeat": 1,
            "version": "v1",
        },
        "turn around": {
            "direction": "left",
            "speed": "normal",
            "repeat": 1,
            "version": "v1",
        },
        "WakaWaka": {
            "direction": "both",
            "speed": "normal",
            "repeat": 1,
            "version": "v1",
        },
        "Hug": {"direction": "", "speed": "normal", "repeat": 1, "version": "v1"},
        "RaiseRightHand": {
            "direction": "",
            "speed": "normal",
            "repeat": 1,
            "version": "v1",
        },
        "PushUp": {
            "direction": "",
            "speed": "normal",
            "repeat": 1,
            "version": "v1",
        },
    }

    def __post_init__(self):
        """
        Set default values for optional fields based on the motion name.
        """
        defaults = self._defaults.get(self.name)
        if defaults is None:
            raise ValueError(f"Unknown motion name {self.name!r}")
        for field_name, default_val in defaults.items():
            if getattr(self, field_name) is None:
                setattr(self, field_name, default_val)


class MoveYansheeConfig(ActionConfig):
    """
    Configuration for Yanshee motion connector.

    Parameters
    ----------
    robot_ip : str
        IP address of the Yanshee robot.
    """

    robot_ip: str = Field(
        default="127.0.0.1",
        description="IP address of the Yanshee robot.",
    )


class MoveYansheeConnector(ActionConnector[MoveYansheeConfig, MoveInput]):
    """
    Connector that sends move commands to a Yanshee robot using the UBTECH YanAPI.
    """

    def __init__(self, config: MoveYansheeConfig):
        """
        Initializes the connector with the given configuration.

        Parameters
        ----------
        config : MoveYansheeConfig
            Configuration for the Yanshee motion connector.
        """
        super().__init__(config)

        self.joysticks = []

        self.vendor_id = ""
        self.product_id = ""
        self.button_previous = None
        self.d_pad_previous = None
        self.rt_previous = 0
        self.lt_previous = 0
        self.gamepad = None

        self.move_speed = 0.7
        self.turn_speed = 0.6
        self.timeout = 8.0

        try:
            robot_ip = self.config.robot_ip
            YanAPI.yan_api_init(robot_ip)
        except Exception as e:
            logging.error(f"Error performing init: {e}")

        try:
            self._send_command(Motion("reset"))
            self._send_command(Motion("wave"))
            logging.info("Robot starting up")
        except Exception as e:
            logging.error(f"Error starting: {e}")

        self.thread_lock = threading.Lock()

    def _send_command(self, motion: Motion):
        """
        Send a motion command to the UBTECH robot.

        Parameters
        ----------
        motion : Motion
            The motion command to send.
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(YanAPI.sync_play_motion, **asdict(motion))
                result = future.result(timeout=self.timeout)
            logging.info("Sent Command %r", asdict(motion))
            if motion.name != "reset":
                return self._send_command(Motion("reset"))
            else:
                return result

        except concurrent.futures.TimeoutError:
            logging.error(
                "Timeout when sending command %r (>%s seconds)",
                asdict(motion),
                self.timeout,
            )
            return False

        except ValueError as err:
            logging.error("Play motion parameter error for %r: %s", asdict(motion), err)
            return False

        except Exception as err:
            logging.error(
                "Unexpected error sending command %r: %s", asdict(motion), err
            )
            return False

    def _execute_command_thread(self, motion: Motion) -> None:
        """
        Execute a motion command in a separate thread.

        Parameters
        ----------
        motion : Motion
            The motion command to execute.
        """
        try:
            self._send_command(motion)

            logging.info(f"UB command {motion.name} executed")

        except Exception as e:
            logging.error(f"Error in command thread {motion.name}: {e}")
        finally:
            self.thread_lock.release()

    def _execute_sport_command_sync(self, motion: Motion) -> None:
        """
        Execute a sport motion command synchronously.

        Parameters
        ----------
        motion : Motion
            The motion command to execute.
        """
        if not self.thread_lock.acquire(blocking=False):
            logging.info("Action already in progress, skipping")
            return

        try:
            thread = threading.Thread(
                target=self._execute_command_thread, args=(motion,), daemon=True
            )
            thread.start()
        except Exception as e:
            logging.error(f"Error executing UB command {motion.name}: {e}")
            self.thread_lock.release()

    async def _execute_sport_command(self, motion: Motion) -> None:
        """
        Execute a sport motion command asynchronously.

        Parameters
        ----------
        motion : Motion
            The motion command to execute.
        """
        if not self.thread_lock.acquire(blocking=False):
            logging.info("Action already in progress, skipping")
            return

        try:
            thread = threading.Thread(
                target=self._execute_command_thread, args=(motion,), daemon=True
            )
            thread.start()
        except Exception as e:
            logging.error(f"Error executing UB command {motion.name}: {e}")
            self.thread_lock.release()

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Process a move action by sending commands to the UBTECH robot.

        Parameters
        ----------
        output_interface : MoveInput
            The MoveInput interface containing the move action.
        """
        if output_interface.action == "wave":
            logging.info("UB command: wave")
            await self._execute_sport_command(Motion("wave"))
        elif output_interface.action == "walk forward":
            logging.info("UB command: walk forward")
            await self._execute_sport_command(
                Motion("walk", direction="forward", repeat=2)
            )
        elif output_interface.action == "walk backward":
            logging.info("UB command: walk backward")
            await self._execute_sport_command(
                Motion("walk", direction="backward", repeat=2)
            )
        elif output_interface.action == "walk left":
            logging.info("UB command: walk left")
            await self._execute_sport_command(
                Motion("walk", direction="left", repeat=2)
            )
        elif output_interface.action == "walk right":
            logging.info("UB command: walk right")
            await self._execute_sport_command(
                Motion("walk", direction="right", repeat=2)
            )
        elif output_interface.action == "turn left":
            logging.info("UB command: turn left")
            await self._execute_sport_command(Motion("turn around", direction="left"))
        elif output_interface.action == "turn right":
            logging.info("UB command: turn right")
            await self._execute_sport_command(Motion("turn around", direction="right"))
        elif output_interface.action == "look left":
            logging.info("UB command: look left")
            await self._execute_sport_command(Motion("head", direction="left"))
        elif output_interface.action == "look right":
            logging.info("UB command: look right")
            await self._execute_sport_command(Motion("head", direction="right"))
        elif output_interface.action == "bow":
            logging.info("UB command: bow")
            await self._execute_sport_command(Motion("bow"))
        elif output_interface.action == "crouch":
            logging.info("UB command: crouch")
            await self._execute_sport_command(Motion("crouch"))
        elif output_interface.action == "come on":
            logging.info("UB command: come on")
            await self._execute_sport_command(Motion("come on"))
        elif output_interface.action == "WakaWaka":
            logging.info("UB command: waka waka")
            await self._execute_sport_command(Motion("WakaWaka"))
        elif output_interface.action == "Hug":
            logging.info("UB command: hug")
            await self._execute_sport_command(Motion("Hug"))
        elif output_interface.action == "RaiseRightHand":
            logging.info("UB command: raise right hand")
            await self._execute_sport_command(Motion("RaiseRightHand"))
        elif output_interface.action == "PushUp":
            logging.info("UB command: push up")
            await self._execute_sport_command(Motion("PushUp"))
        elif output_interface.action == "reset":
            logging.info("UB command: stand still")
        else:
            logging.info(f"Unknown move type: {output_interface.action}")

        logging.info(f"SendThisToUB: {output_interface.action}")

    def tick(self) -> None:
        """
        Periodic tick function to maintain connection.
        """
        time.sleep(0.1)
