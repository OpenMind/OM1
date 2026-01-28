import logging
import threading
import time
from typing import Optional

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.move_game_controller.interface import IDLEInput
from providers.odom_provider import OdomProvider, RobotState
from providers.unitree_go2_state_provider import UnitreeGo2StateProvider
from unitree.unitree_sdk2py.go2.sport.sport_client import SportClient
from zenoh_msgs import AudioStatus, open_zenoh_session

try:
    import hid
except ImportError:
    logging.warning(
        "HID library not found. Please install the HIDAPI library to use this plugin."
    )
    hid = None


class Go2GameControllerConfig(ActionConfig):
    """
    Configuration for Go2GameController connector.

    Parameters
    ----------
    speed_x : float
        Movement speed in the x direction (m/s).
    speed_yaw : float
        Rotation speed around the yaw axis (rad/s).
    yaw_correction : float
        Yaw correction factor.
    lateral_correction : float
        Lateral correction factor.
    unitree_ethernet : Optional[str]
        Ethernet channel for Unitree Go2 odometry.
    """

    speed_x: float = Field(
        default=0.9,
        description="Movement speed in the x direction (m/s).",
    )
    speed_yaw: float = Field(
        default=0.6,
        description="Rotation speed around the yaw axis (rad/s).",
    )
    yaw_correction: float = Field(
        default=0.0,
        description="Yaw correction factor.",
    )
    lateral_correction: float = Field(
        default=0.0,
        description="Lateral correction factor.",
    )
    unitree_ethernet: Optional[str] = Field(
        default=None,
        description="Ethernet channel for Unitree Go2 odometry.",
    )


class Go2GameControllerConnector(ActionConnector[Go2GameControllerConfig, IDLEInput]):
    """
    Game controller for Unitree Go2 robots.

    NOTE: This connector has been deprecated. OM1 Orchestrator docker automatically supports game controller.
    """

    def __init__(self, config: Go2GameControllerConfig):
        """
        Initialize the game controller connector.

        Parameters
        ----------
        config : Go2GameControllerConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        self.config = config

        # Movement speed m/s and rad/s
        self.move_speed = self.config.speed_x
        self.turn_speed = self.config.speed_yaw
        self.yaw_correction = self.config.yaw_correction
        self.lateral_correction = self.config.lateral_correction

        self.topic = "robot/status/audio"
        self.session = None
        try:
            self.session = open_zenoh_session()
            self.session.declare_subscriber(self.topic, self.zenoh_audio_message)
            logging.info("Game Controller Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Game Controller Zenoh client: {e}")

        self.gamepad = None
        self.sony_dualsense = False
        self.sony_edge = False
        self.xbox = False
        self._init_controller()

        if self.gamepad is None:
            logging.warning("Game controller not found")

        self.sport_client = None
        try:
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(10.0)
            self.sport_client.Init()
            self.sport_client.StopMove()
            self.sport_client.Move(0.05, 0, 0)
            logging.info("Game controller Unitree sport client initialized")
        except Exception as e:
            self.sport_client = None
            logging.error(
                f"Error initializing game controller Unitree sport client: {e}"
            )

        # Pad buttons
        self.rt_previous = 0
        self.lt_previous = 0
        self.d_pad_previous = 0
        self.button_previous = 0

        self.lt_value = 0
        self.rt_value = 0
        self.d_pad_value = 0
        self.button_value = 0

        self.RTLT_moving = False

        unitree_ethernet = self.config.unitree_ethernet
        self.odom = OdomProvider(channel=unitree_ethernet)
        self.unitree_state_provider = UnitreeGo2StateProvider()

        self.thread_lock = threading.Lock()

    def zenoh_audio_message(self, data: zenoh.Sample) -> None:
        """
        Callback function to handle incoming Zenoh audio messages.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

    def _init_controller(self) -> None:
        """
        Initialize or reinitialize the game controller.
        """
        self.sony_dualsense = False
        self.xbox = False
        self.sony_edge = False

        if hid is not None:
            for device in hid.enumerate():
                logging.debug(f"device {device['product_string']}")
                if "Xbox Wireless Controller" in device["product_string"]:
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    try:
                        self.gamepad = hid.Device(vendor_id, product_id)
                        logging.info(
                            f"Connected {device['product_string']} {vendor_id} {product_id}"
                        )
                        self.xbox = True
                        break
                    except Exception as e:
                        logging.error(f"Failed to connect to Xbox controller: {e}")
                        continue
                if "DualSense Wireless Controller" in device["product_string"]:
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    try:
                        self.gamepad = hid.Device(vendor_id, product_id)
                        logging.info(
                            f"Connected {device['product_string']} {vendor_id} {product_id}"
                        )
                        self.sony_dualsense = True
                        break
                    except Exception as e:
                        logging.error(f"Failed to connect to DualSense controller: {e}")
                        continue
                if "DualSense Edge Wireless Controller" in device["product_string"]:
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    try:
                        self.gamepad = hid.Device(vendor_id, product_id)
                        logging.info(
                            f"Connected {device['product_string']} {vendor_id} {product_id}"
                        )
                        self.sony_edge = True
                        break
                    except Exception as e:
                        logging.error(
                            f"Failed to connect to DualSense Edge controller: {e}"
                        )
                        continue

    def _execute_command_thread(self, command: str) -> None:
        """
        Execute a Unitree command in a separate thread.

        Parameters
        ----------
        command : str
            The command to execute.
        """
        try:
            if (
                command == "StandUp"
                and self.odom.position["body_attitude"] is RobotState.STANDING
            ):
                logging.info("Already standing, skipping command")
                return
            elif (
                command == "StandDown"
                and self.odom.position["body_attitude"] is RobotState.SITTING
            ):
                logging.info("Already sitting, skipping command")
                return

            if self.unitree_state_provider.state_code == 1002:
                if self.sport_client:
                    logging.info("Robot is in jointLock state - issuing BalanceStand()")
                    self.sport_client.BalanceStand()
                    self.sport_client.Move(0.05, 0, 0)

            code = getattr(self.sport_client, command)()
            logging.info(f"Unitree command {command} executed with code {code}")

            if self.unitree_state_provider.state_code == 1002:
                if self.sport_client:
                    logging.info("Robot is in jointLock state - issuing BalanceStand()")
                    self.sport_client.BalanceStand()
                    self.sport_client.Move(0.05, 0, 0)

        except Exception as e:
            logging.error(f"Error in command thread {command}: {e}")
        finally:
            self.thread_lock.release()

    def _execute_sport_command_sync(self, command: str) -> None:

        logging.debug(f"_execute_sport_command_sync({command})")

        if self.sport_client is None:
            logging.debug("No sport client, skipping")
            return

        if not self.thread_lock.acquire(blocking=False):
            logging.debug("Action already in progress, skipping")
            return

        try:
            thread = threading.Thread(
                target=self._execute_command_thread, args=(command,), daemon=True
            )
            thread.start()
        except Exception as e:
            logging.error(f"Error executing Unitree command {command}: {e}")
            self.thread_lock.release()

    async def connect(self, output_interface: IDLEInput) -> None:
        """
        Connect to actions from LLM.

        Parameters
        ----------
        output_interface : IDLEInput
            The output interface to connect to.

        Returns
        -------
        None
        """
        pass

    def _move_robot(self, vx, vy, vturn=0.0) -> None:
        """
        Move the robot using the sport client.

        Parameters
        ----------
        vx : float
            The speed in the x direction.
        vy : float
            The speed in the y direction.
        vturn : float
            The speed of rotation.

        Returns
        -------
        None
        """
        logging.info(f"GAME _move_robot: vx={vx}, vy={vy}, vturn={vturn}")

        if not self.sport_client:
            return

        if self.odom.position["body_attitude"] is not RobotState.STANDING:
            logging.info("self.sport_client.Move blocked - dog is sitting")
            return

        if self.unitree_state_provider.state == "jointLock":
            self.sport_client.BalanceStand()
            self.sport_client.Move(0.05, 0, 0)

        try:
            logging.info(f"self.sport_client.Move: vx={vx}, vy={vy}, vturn={vturn}")
            self.sport_client.Move(vx, vy, vturn)
        except Exception as e:
            logging.error(f"Error moving robot: {e}")

    def tick(self) -> None:
        """
        Tick function is called periodically to check for input from the Xbox controller.
        It reads the input from the controller and sends the corresponding commands to the robot.

        Returns
        -------
        None
        """
        time.sleep(0.05)
        logging.debug("Gamepad tick")

        data = None

        # Attempt reconnection if no gamepad is currently attached
        if self.gamepad is None and hid is not None:
            logging.warning("Controller disconnected - will try to reconnect")
            self._init_controller()

        if self.gamepad:
            try:
                # try to read USB data, and if there is nothing, timeout
                data = list(self.gamepad.read(64, timeout=50))
            except Exception as e:
                logging.warning(f"Controller disconnected: {e}")
                self.gamepad = None

        # special case for no data if the D-pad or LT and RT is kept pressed
        if data is None or len(data) == 0:
            if self.rt_previous > 0 or self.lt_previous > 0:
                # we always execute the left turn first
                if self.lt_previous > 0:
                    logging.info(
                        "Left Trigger is kept pressed - Counter-clockwise rotation"
                    )
                    self._move_robot(0.0, 0.0, self.turn_speed)
                    return
                if self.rt_previous > 0:
                    logging.info("Right Trigger is kept pressed - Clockwise rotation")
                    self._move_robot(0.0, 0.0, -self.turn_speed)
                    return

            if self.d_pad_previous and self.d_pad_previous > 0:
                if self.d_pad_previous == 1:  # Up
                    logging.info("D-pad UP - Moving forward")
                    self._move_robot(
                        self.move_speed, self.lateral_correction, self.yaw_correction
                    )
                elif self.d_pad_previous == 5:  # Down
                    logging.info("D-pad DOWN - Moving backward")
                    self._move_robot(
                        -self.move_speed, -self.lateral_correction, -self.yaw_correction
                    )
                elif self.d_pad_previous == 7:  # Left
                    logging.info("D-pad LEFT - Moving left")
                    self._move_robot(0.0, self.move_speed)
                elif self.d_pad_previous == 3:  # Right
                    logging.info("D-pad RIGHT - Moving right")
                    self._move_robot(0.0, -self.move_speed)

            return

        if data and len(data) > 0:

            logging.debug(f"Gamepad data: {data}")

            # deal with the different mappings
            if self.xbox:
                logging.debug(f"Gamepad data Xbox: {data}")
                self.lt_value = data[9]  # Left Trigger
                self.rt_value = data[11]  # Right Trigger
                self.d_pad_value = data[13]
                self.button_value = data[14]
            elif self.sony_dualsense or self.sony_edge:
                logging.debug(f"Gamepad data Sony: {data}")

                multi = 0

                if len(data) > 10:
                    logging.debug("Gamepad data Sony length > 10")
                    self.lt_value = data[5]  # Left Trigger
                    self.rt_value = data[6]  # Right Trigger
                    multi = data[8]
                elif len(data) == 10:
                    logging.debug("Gamepad data Sony length == 10")
                    self.lt_value = data[8]  # Left Trigger
                    self.rt_value = data[9]  # Right Trigger
                    multi = data[5]

                if multi == 8:
                    self.d_pad_value = 0
                    self.button_value = 0
                elif multi == 0:
                    self.d_pad_value = 1  # up
                    self.button_value = 0
                elif multi == 6:
                    self.d_pad_value = 7  # left
                    self.button_value = 0
                elif multi == 2:
                    self.d_pad_value = 3  # right
                    self.button_value = 0
                elif multi == 4:
                    self.d_pad_value = 5  # back
                    self.button_value = 0
                elif multi == 40:
                    # "A" aka X button
                    self.d_pad_value = 0
                    self.button_value = 1
                elif multi == 72:
                    # "B" aka 0 button
                    self.d_pad_value = 0
                    self.button_value = 2

                logging.debug(f"Gamepad data lt: {self.lt_value}")
                logging.debug(f"Gamepad data rt: {self.rt_value}")
                logging.debug(f"Gamepad data multi: {multi}")

            move_triggered_RTLT = False
            move_triggered_dpad = False

            # Trigger values range from 0 to 255
            # Check if the triggers have changed significantly
            # rt_changed = abs(self.rt_value - self.rt_previous) > 5
            # lt_changed = abs(self.lt_value - self.lt_previous) > 5

            # Normalize trigger values from 0-255 to 0-1.0
            rt = self.rt_value / 255.0
            lt = self.lt_value / 255.0

            previous_RTLT = self.lt_previous + self.rt_previous

            # Right Trigger - clockwise rotation
            if rt > 0.8 and rt > lt:
                move_triggered_RTLT = True
                self._move_robot(0.0, 0.0, -self.turn_speed)
            # Left Trigger - counter-clockwise rotation
            elif lt > 0.8 and lt > rt:
                move_triggered_RTLT = True
                self._move_robot(0.0, 0.0, self.turn_speed)
            # Both triggers released or below threshold
            elif (rt <= 0.8 and lt <= 0.8) and previous_RTLT > 0.8:
                # we WERE moving, but we just let go of one or both triggers
                move_triggered_RTLT = True
                logging.debug("Triggers released - Stopping rotation")

            # Update previous Trigger values
            self.rt_previous = rt
            self.lt_previous = lt

            if move_triggered_RTLT:
                # update the previous value of the button
                self.button_previous = self.button_value
                # and return, since we just issued a action command in this tick
                return

            # Control robot movement based on D-pad
            logging.debug(f"Gamepad DPAD: {self.d_pad_value}")

            if self.d_pad_value == 1:  # Up
                logging.info("D-pad UP - Moving forward")
                move_triggered_dpad = True
                self._move_robot(
                    self.move_speed, self.lateral_correction, self.yaw_correction
                )
            elif self.d_pad_value == 5:  # Down
                logging.info("D-pad DOWN - Moving backward")
                move_triggered_dpad = True
                self._move_robot(
                    -self.move_speed, -self.lateral_correction, -self.yaw_correction
                )
            elif self.d_pad_value == 7:  # Left
                logging.info("D-pad LEFT - Moving left")
                move_triggered_dpad = True
                self._move_robot(0.0, self.move_speed)
            elif self.d_pad_value == 3:  # Right
                logging.info("D-pad RIGHT - Moving right")
                move_triggered_dpad = True
                self._move_robot(0.0, -self.move_speed)
            elif self.d_pad_previous > 0 and self.d_pad_value == 0:
                # User just released DPAD
                logging.info("D-pad released - Stopping movement")
                move_triggered_dpad = True
                self._move_robot(0.0, 0.0)

            # update the value of d_pad_previous
            self.d_pad_previous = self.d_pad_value

            if move_triggered_dpad:
                self.button_previous = self.button_value
                # return, since we just issued a DPAD move command in this tick
                return

            # logging.debug(f"Gamepad button value {button_value}")

            if self.button_previous == 0 and self.button_value > 0:

                # logging.debug(f"Gamepad button pressed")

                # We need this logic because when the user presses a button
                # the gamepad sends a 'press' indication numerous times
                # for several hundred ms, creating numerous
                # duplicated movement commands with a single button press.
                # To prevent this, which would freeze/crash the robot,
                # we only act when the button state changes from 0 to > 0
                # This is basically a software button debounce

                # button A
                if self.button_value == 1:
                    logging.info("Controller unitree: stand_up")
                    self._execute_sport_command_sync("StandUp")
                # button B
                elif self.button_value == 2:
                    logging.info("Controller unitree: lay_down")
                    self._execute_sport_command_sync("StandDown")
                # # X button
                # elif button_value == 8:
                #     self._execute_sport_command_sync("Hello")
                #     logging.info("Controller unitree: say_hello")
                # # Y button
                # elif button_value == 16:
                #     self._execute_sport_command_sync("Stretch")
                #     logging.info("Controller unitree: stretch")

                logging.info(f"Gamepad button depressed edge {self.button_value}")

            # update the value of button_previous
            self.button_previous = self.button_value
