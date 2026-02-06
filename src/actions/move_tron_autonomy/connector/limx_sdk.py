import logging
import math
import random
from queue import Queue
from typing import List, Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector, MoveCommand
from actions.move_tron_autonomy.interface import MoveInput
from providers.simple_paths_provider import SimplePathsProvider
from providers.tron_odom_provider import RobotState, TronOdomProvider
from zenoh_msgs import geometry_msgs, open_zenoh_session


class MoveTronZenohConfig(ActionConfig):
    """
    Configuration for Tron Zenoh connector.

    Parameters
    ----------
    odom_topic : str
        Zenoh topic for odometry data.
    cmd_vel_topic : str
        Zenoh topic for velocity commands.
    """

    odom_topic: str = Field(
        default="odom",
        description="Zenoh topic for odometry data.",
    )
    cmd_vel_topic: str = Field(
        default="cmd_vel",
        description="Zenoh topic for velocity commands.",
    )


class MoveTronZenohConnector(ActionConnector[MoveTronZenohConfig, MoveInput]):
    """
    Zenoh connector for the Move Tron autonomy action.
    Uses Zenoh to publish cmd_vel commands and receive odom data from limx-sdk.
    """

    def __init__(self, config: MoveTronZenohConfig):
        """
        Initialize the Zenoh connector for Tron robot.

        Parameters
        ----------
        config : MoveTronZenohConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        # Movement parameters
        self.move_speed = 0.25
        self.turn_speed = 0.35
        self.angle_tolerance = 5.0  # degrees
        self.distance_tolerance = 0.05  # meters
        self.pending_movements: Queue[Optional[MoveCommand]] = Queue()
        self.movement_attempts = 0
        self.movement_attempt_limit = 15
        self.gap_previous = 0

        self.session = None

        odom_topic = self.config.odom_topic
        self.cmd_vel_topic = self.config.cmd_vel_topic

        try:
            self.session = open_zenoh_session()
            logging.info(f"Tron Zenoh move client opened {self.session}")
        except Exception as e:
            logging.error(f"Error opening Zenoh client for Tron: {e}")

        self.path_provider = SimplePathsProvider()
        self.odom = TronOdomProvider(topic=odom_topic)

        logging.info(f"Tron Autonomy Odom Provider: {self.odom}")
        logging.info(f"Tron Autonomy cmd_vel topic: {self.cmd_vel_topic}")

    def _move_robot(self, vx: float, vy: float, vyaw: float) -> None:
        """
        Generate movement commands via Zenoh cmd_vel topic.

        Parameters
        ----------
        vx : float
            Linear velocity in the x direction (m/s).
        vy : float
            Linear velocity in the y direction (m/s).
        vyaw : float
            Angular velocity around the z axis (rad/s).
        """
        logging.debug(f"move: vx={vx}, vy={vy}, vyaw={vyaw}")

        if self.session is None:
            logging.info("No open Zenoh session, returning")
            return

        if self.odom.position["body_attitude"] != RobotState.STANDING:
            logging.info("Cannot move - robot is not standing")
            return

        logging.debug(f"Pub twist: vx={vx}, vy={vy}, vyaw={vyaw}")
        t = geometry_msgs.Twist(
            linear=geometry_msgs.Vector3(x=float(vx), y=float(vy), z=0.0),
            angular=geometry_msgs.Vector3(x=0.0, y=0.0, z=float(vyaw)),
        )
        self.session.put(self.cmd_vel_topic, t.serialize())

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect to the output interface and process the AI movement command.

        Parameters
        ----------
        output_interface : MoveInput
            The output interface containing the AI movement command.
        """
        logging.info(f"Tron AI command.connect: {output_interface.action}")

        if self.odom.position["moving"]:
            logging.info("Disregard new AI movement command - robot is already moving")
            return

        if self.pending_movements.qsize() > 0:
            logging.info("Movement in progress: disregarding new AI command")
            return

        if self.odom.position["odom_x"] == 0.0:
            # this value is never precisely zero EXCEPT while
            # booting and waiting for data to arrive
            logging.info("Waiting for location data")
            return

        # Process movement commands with lidar safety checks
        movement_map = {
            "turn left": self._process_turn_left,
            "turn right": self._process_turn_right,
            "move forwards": self._process_move_forward,
            "move back": self._process_move_back,
            "stand still": lambda: logging.info("AI movement command: stand still"),
        }

        handler = movement_map.get(output_interface.action)
        if handler:
            handler()
        else:
            logging.info(f"AI movement command unknown: {output_interface.action}")

    def clean_abort(self) -> None:
        """
        Cleanly abort current movement and reset state.
        """
        self.movement_attempts = 0
        if not self.pending_movements.empty():
            self.pending_movements.get()

    def tick(self) -> None:
        """
        Process the AI motion tick.
        """
        logging.debug("Tron AI Motion Tick")

        if self.odom is None:
            logging.info("Waiting for odom data = self.odom is None")
            self.sleep(0.5)
            return

        if self.odom.position["odom_x"] == 0.0:
            # this value is never precisely zero except while
            # booting and waiting for data to arrive
            logging.info("Waiting for odom data, x == 0.0")
            self.sleep(0.5)
            return

        if self.odom.position["body_attitude"] != RobotState.STANDING:
            logging.info("Cannot move - robot is not standing")
            self.sleep(0.5)
            return

        # if we got to this point, we have good data and we are able to
        # safely proceed
        target: List[MoveCommand] = list(self.pending_movements.queue)

        if len(target) > 0:

            current_target = target[0]

            logging.info(
                f"Target: {current_target} current yaw: {self.odom.position['odom_yaw_m180_p180']}"
            )

            if self.movement_attempts > self.movement_attempt_limit:
                # abort - we are not converging
                self.clean_abort()
                logging.info(
                    f"TIMEOUT - not converging after {self.movement_attempt_limit} attempts - StopMove()"
                )
                return

            goal_dx = current_target.dx
            goal_yaw = current_target.yaw

            # Phase 1: Turn to face the target direction
            if not current_target.turn_complete:
                gap = self._calculate_angle_gap(
                    -1 * self.odom.position["odom_yaw_m180_p180"], goal_yaw
                )
                logging.info(f"Phase 1 - Turning remaining GAP: {gap}DEG")

                progress = round(abs(self.gap_previous - gap), 2)
                self.gap_previous = gap
                if self.movement_attempts > 0:
                    logging.info(f"Phase 1 - Turn GAP delta: {progress}DEG")

                if abs(gap) > 10.0:
                    logging.debug("Phase 1 - Gap is big, using large displacements")
                    self.movement_attempts += 1
                    if not self._execute_turn(gap):
                        self.clean_abort()
                        return
                elif abs(gap) > self.angle_tolerance and abs(gap) <= 10.0:
                    logging.debug("Phase 1 - Gap is decreasing, using smaller steps")
                    self.movement_attempts += 1
                    # rotate only because we are so close
                    # no need to check barriers because we are just performing small rotations
                    if gap > 0:
                        self._move_robot(0, 0, 0.2)
                    elif gap < 0:
                        self._move_robot(0, 0, -0.2)
                elif abs(gap) <= self.angle_tolerance:
                    logging.info("Phase 1 - Turn completed, starting movement")
                    current_target.turn_complete = True
                    self.gap_previous = 0

            else:
                # Phase 2: Move towards the target position, if needed
                if goal_dx == 0:
                    logging.info("No movement required, processing next AI command")
                    self.clean_abort()
                    return

                s_x = current_target.start_x
                s_y = current_target.start_y
                speed = current_target.speed

                distance_traveled = math.sqrt(
                    (self.odom.position["odom_x"] - s_x) ** 2
                    + (self.odom.position["odom_y"] - s_y) ** 2
                )
                gap = round(abs(goal_dx - distance_traveled), 2)
                progress = round(abs(self.gap_previous - gap), 2)
                self.gap_previous = gap

                if self.movement_attempts > 0:
                    logging.info(f"Phase 2 - Forward/retreat GAP delta: {progress}m")

                fb = 0
                if goal_dx > 0:
                    if 4 not in self.path_provider.advance:
                        logging.warning("Cannot advance due to barrier")
                        self.clean_abort()
                        return
                    fb = 1

                if goal_dx < 0:
                    if not self.path_provider.retreat:
                        logging.warning("Cannot retreat due to barrier")
                        self.clean_abort()
                        return
                    fb = -1

                if gap > self.distance_tolerance:
                    self.movement_attempts += 1
                    if distance_traveled < abs(goal_dx):
                        logging.info(f"Phase 2 - Keep moving. Remaining: {gap}m ")
                        self._move_robot(fb * speed, 0.0, 0.0)
                    elif distance_traveled > abs(goal_dx):
                        logging.debug(
                            f"Phase 2 - OVERSHOOT: move other way. Remaining: {gap}m"
                        )
                        self._move_robot(-1 * fb * 0.15, 0.0, 0.0)
                else:
                    logging.info(
                        "Phase 2 - Movement completed normally, processing next AI command"
                    )
                    self.clean_abort()

        self.sleep(0.1)

    def _process_turn_left(self):
        """
        Process turn left command with safety check.
        """
        if not self.path_provider.turn_left:
            logging.warning("Cannot turn left due to barrier")
            return

        path = random.choice(self.path_provider.turn_left)
        path_angle = self.path_provider.path_angles[path]

        target_yaw = self._normalize_angle(
            -1 * self.odom.position["odom_yaw_m180_p180"] + path_angle
        )
        self.pending_movements.put(
            MoveCommand(
                dx=0.5,
                yaw=round(target_yaw, 2),
                start_x=round(self.odom.position["odom_x"], 2),
                start_y=round(self.odom.position["odom_y"], 2),
                turn_complete=False,
            )
        )

    def _process_turn_right(self):
        """
        Process turn right command with safety check.
        """
        if not self.path_provider.turn_right:
            logging.warning("Cannot turn right due to barrier")
            return

        path = random.choice(self.path_provider.turn_right)
        path_angle = self.path_provider.path_angles[path]

        target_yaw = self._normalize_angle(
            -1 * self.odom.position["odom_yaw_m180_p180"] + path_angle
        )
        self.pending_movements.put(
            MoveCommand(
                dx=0.5,
                yaw=round(target_yaw, 2),
                start_x=round(self.odom.position["odom_x"], 2),
                start_y=round(self.odom.position["odom_y"], 2),
                turn_complete=False,
            )
        )

    def _process_move_forward(self):
        """
        Process move forward command with safety check.
        """
        if not self.path_provider.advance:
            logging.warning("Cannot advance due to barrier")
            return

        path = random.choice(self.path_provider.advance)
        path_angle = self.path_provider.path_angles[path]

        target_yaw = self._normalize_angle(
            -1 * self.odom.position["odom_yaw_m180_p180"] + path_angle
        )
        self.pending_movements.put(
            MoveCommand(
                dx=0.5,
                yaw=target_yaw,
                start_x=round(self.odom.position["odom_x"], 2),
                start_y=round(self.odom.position["odom_y"], 2),
                turn_complete=True if path_angle == 0 else False,
            )
        )

    def _process_move_back(self):
        """
        Process move back command with safety check.
        """
        if not self.path_provider.retreat:
            logging.warning("Cannot retreat due to barrier")
            return

        self.pending_movements.put(
            MoveCommand(
                dx=-0.5,
                yaw=0.0,
                start_x=round(self.odom.position["odom_x"], 2),
                start_y=round(self.odom.position["odom_y"], 2),
                turn_complete=True,
                speed=0.25,
            )
        )

    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-180, 180] range.

        Parameters
        ----------
        angle : float
            Angle in degrees to normalize.

        Returns
        -------
        float
            Normalized angle in degrees within the range [-180, 180].
        """
        if angle < -180:
            angle += 360.0
        elif angle > 180:
            angle -= 360.0
        return angle

    def _calculate_angle_gap(self, current: float, target: float) -> float:
        """
        Calculate shortest angular distance between two angles.

        Parameters
        ----------
        current : float
            Current angle in degrees.
        target : float
            Target angle in degrees.

        Returns
        -------
        float
            Shortest angular distance in degrees, rounded to 2 decimal places.
        """
        gap = current - target
        if gap > 180.0:
            gap -= 360.0
        elif gap < -180.0:
            gap += 360.0
        return round(gap, 2)

    def _execute_turn(self, gap: float) -> bool:
        """
        Execute turn based on gap direction and lidar constraints.

        Parameters
        ----------
        gap : float
            The angle gap in degrees to turn.

        Returns
        -------
        bool
            True if the turn was executed successfully, False if blocked by a barrier.
        """
        if gap > 0:  # Turn left
            if not self.path_provider.turn_left:
                logging.warning("Cannot turn left due to barrier")
                return False
            sharpness = min(self.path_provider.turn_left)
            self._move_robot(sharpness * 0.15, 0, self.turn_speed)
        else:  # Turn right
            if not self.path_provider.turn_right:
                logging.warning("Cannot turn right due to barrier")
                return False
            sharpness = 8 - max(self.path_provider.turn_right)
            self._move_robot(sharpness * 0.15, 0, -self.turn_speed)
        return True
