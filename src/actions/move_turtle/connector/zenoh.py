import logging
import math
import random
from queue import Queue
from typing import List, Optional

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector, MoveCommand
from actions.move_turtle.interface import MoveInput
from providers.turtlebot4_odom_provider import TurtleBot4OdomProvider
from providers.turtlebot4_rplidar_provider import TurtleBot4RPLidarProvider
from zenoh_msgs import geometry_msgs, open_zenoh_session, sensor_msgs


class MoveZenohConfig(ActionConfig):
    """
    Configuration for Zenoh connector.

    Parameters
    ----------
    URID : Optional[str]
        URID for Zenoh topics.
    """

    URID: Optional[str] = Field(
        default=None,
        description="URID for Zenoh topics.",
    )


class MoveZenohConnector(ActionConnector[MoveZenohConfig, MoveInput]):
    """
    Zenoh connector for the Move Turtlebot4 action.
    """

    def __init__(self, config: MoveZenohConfig):
        """
        Initialize the Zenoh connector.

        Parameters
        ----------
        config : MoveZenohConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        self.turn_speed = 0.8
        self.angle_tolerance = 5.0
        self.distance_tolerance = 0.05  # m

        self.pending_movements: Queue[Optional[MoveCommand]] = Queue()

        self.hazard = None
        self.emergency = None

        self.session = None

        URID = self.config.URID

        if URID is None:
            logging.warning("Aborting TurtleBot4 Move system, no URID provided")
            return
        else:
            logging.info(f"TurtleBot4 Move system is using URID: {URID}")

        self.cmd_vel = f"{URID}/c3/cmd_vel"

        try:
            self.session = open_zenoh_session()
            logging.info(f"Zenoh move client opened {self.session}")
            logging.info(f"TurtleBot4 hazard listener starting with URID: {URID}")
            self.session.declare_subscriber(
                f"{URID}/c3/hazard_detection", self.listen_hazard
            )
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

        self.lidar = TurtleBot4RPLidarProvider()
        self.odom = TurtleBot4OdomProvider(URID=URID)

    def listen_hazard(self, data: zenoh.Sample) -> None:
        """
        Callback for Zenoh hazard detection messages.
        This method is called when a hazard detection message is received.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing the hazard detection data.
        """
        self.hazard = sensor_msgs.HazardDetectionVector.deserialize(
            data.payload.to_bytes()
        )
        if (
            self.hazard is not None
            and self.hazard.detections
            and len(self.hazard.detections) > 0  # type: ignore
        ):
            for haz in self.hazard.detections:  # type: ignore
                if haz.type == 1:
                    logging.info(
                        f"Hazard Type:{haz.type} direction:{haz.header.frame_id}"
                    )
                    if "right" in haz.header.frame_id:
                        self.hazard = "TURN_LEFT"
                    elif "left" in haz.header.frame_id:
                        self.hazard = "TURN_RIGHT"
                    elif "center" in haz.header.frame_id:
                        if random.randint(1, 2) == 1:
                            self.hazard = "TURN_LEFT"
                        else:
                            self.hazard = "TURN_RIGHT"
                    logging.info(f"Hazard decision: {self.hazard}")

    def move(self, vx: float, vyaw: float) -> None:
        """
        Generate movement commands.

        Parameters
        ----------
        vx : float
            Linear velocity in the x direction.
        vyaw : float
            Angular velocity around the z axis.
        """
        logging.debug(f"move: {vx} - {vyaw}")

        if self.session is None:
            logging.info("No open Zenoh session, returning")
            return

        logging.debug(f"Pub twist: {vx} - {vyaw}")
        t = geometry_msgs.Twist(
            linear=geometry_msgs.Vector3(x=float(vx), y=0.0, z=0.0),
            angular=geometry_msgs.Vector3(x=0.0, y=0.0, z=float(vyaw)),
        )
        self.session.put(self.cmd_vel, t.serialize())

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect to the output interface and process the move action.

        Parameters
        ----------
        output_interface : MoveInput
            The output interface for the move action.
        """
        logging.info(f"AI motion command: {output_interface.action}")

        if self.pending_movements.qsize() > 0:
            logging.info("Movement in progress: disregarding new AI command")
            return

        if self.emergency:
            logging.info("Avoiding barrier: disregarding new AI command")
            return

        if self.odom.x == 0.0:
            # this value is never precisely zero EXCEPT while
            # booting and waiting for data to arrive
            logging.info("Waiting for location data")
            return

        # reconfirm possible paths
        # this is needed due to the 2s latency of the LLMs
        possible_paths = self.lidar.valid_paths

        advance_danger = True
        retreat_danger = True

        if possible_paths:
            logging.info(f"Action - Valid paths: {possible_paths}")
            if 4 in possible_paths:
                advance_danger = False
            if 9 in possible_paths:
                retreat_danger = False

        if output_interface.action == "turn left":
            # turn 90 Deg to the left (CCW)
            target_yaw = self.odom.odom_yaw_m180_p180 - 30.0
            if target_yaw <= -180:
                target_yaw += 360.0
            self.pending_movements.put(MoveCommand(dx=0.0, yaw=target_yaw))
        elif output_interface.action == "turn right":
            # turn 90 Deg to the right (CW)
            target_yaw = self.odom.odom_yaw_m180_p180 + 30.0
            if target_yaw >= 180.0:
                target_yaw -= 360.0
            self.pending_movements.put(MoveCommand(dx=0.0, yaw=target_yaw))
        elif output_interface.action == "move forwards":
            if advance_danger:
                return
            self.pending_movements.put(
                MoveCommand(
                    dx=0.5,
                    yaw=0.0,
                    start_x=self.odom.x,
                    start_y=self.odom.y,
                )
            )
        elif output_interface.action == "move back":
            if retreat_danger:
                return
            self.pending_movements.put(
                MoveCommand(
                    dx=-0.5,
                    yaw=0.0,
                    start_x=self.odom.x,
                    start_y=self.odom.y,
                )
            )
        elif output_interface.action == "stand still":
            logging.info(f"AI movement command: {output_interface.action}")
            # do nothing
        else:
            logging.info(f"AI movement command unknown: {output_interface.action}")

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

    def clean_abort(self) -> None:
        """
        Cleanly abort current movement and reset state.
        """
        self.movement_attempts = 0
        if not self.pending_movements.empty():
            self.pending_movements.get()

    def tick(self) -> None:
        """
        Periodic tick to process movement commands.
        """
        self.sleep(0.1)

        logging.debug("Move tick")

        if self.odom.x == 0.0:
            # this value is never precisely zero except while
            # booting and waiting for data to arrive
            logging.info("Waiting for odom data")
            self.sleep(0.5)
            return

        # physical collision event ALWAYS takes precedence
        if self.hazard is not None:
            if self.hazard == "TURN_RIGHT":
                target_yaw = self.odom.odom_yaw_m180_p180 + 100.0
                if target_yaw >= 180.0:
                    target_yaw -= 360.0
                self.emergency = target_yaw
            elif self.hazard == "TURN_LEFT":
                target_yaw = self.odom.odom_yaw_m180_p180 - 100.0
                if target_yaw <= -180:
                    target_yaw += 360.0
                self.emergency = target_yaw
            else:
                logging.error(f"Cannot parse self.hazard: {self.hazard}")

            # clear the hazard flag
            self.hazard = None
            logging.info(f"Should have non-zero avoidance yaw: {self.emergency}")

        if self.emergency:

            # target = self.emergency
            target = [MoveCommand(dx=0.0, yaw=self.emergency)]
            logging.info(f"Emergency target: {target}")

            gap = self.odom.odom_yaw_m180_p180 - target[0].yaw
            if gap > 180.0:
                gap -= 360.0
            elif gap < -180.0:
                gap += 360.0

            """
            gap is a SIGNED value indicating:
                * the direction to turn to get to goal, and
                * the magnitude remaining to turn

            a mathematically equivalent way to do this is

            a = targetA - sourceA
            a = (a + 180) % 360 - 180
            where mod = (a, n) -> a - floor(a/n) * n
            """
            logging.info(f"GAP: {gap}")
            if abs(gap) > 10.0:
                logging.debug("gap is big, using large displacements")
                if gap > 0:
                    self.move(0.0, 0.3)
                elif gap < 0:
                    self.move(0.0, -0.3)
            elif abs(gap) > self.angle_tolerance and abs(gap) <= 10.0:
                logging.debug("gap is getting smaller, using smaller steps")
                if gap > 0:
                    self.move(0.0, 0.1)
                elif gap < 0:
                    self.move(0.0, -0.1)
            elif abs(gap) <= self.angle_tolerance:
                logging.info("avoidance motion completed, clear emergency")
                self.emergency = None

            # when there is a hazard, focus on clearing it
            return

        # if we got to this point, we have good data and there is hard wall
        # touch emergency

        target: List[MoveCommand] = list(self.pending_movements.queue)

        if len(target) > 0:

            current_target = target[0]

            logging.debug(
                f"Target: {current_target} current yaw: {self.odom.odom_yaw_m180_p180}"
            )

            goal_dx = current_target.dx
            goal_yaw = current_target.yaw

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
                        self.move(0, 0.2)
                    elif gap < 0:
                        self.move(0, -0.2)
                elif abs(gap) <= self.angle_tolerance:
                    logging.info("Phase 1 - Turn completed, starting movement")
                    current_target.turn_complete = True
                    self.gap_previous = 0
            else:
                if goal_dx == 0:
                    logging.info("No movement required, processing next AI command")
                    self.clean_abort()
                    return

                # reconfirm possible paths
                pp = self.lidar.valid_paths

                logging.debug(f"Action - Valid paths: {pp}")

                s_x = current_target.start_x
                s_y = current_target.start_y
                distance_traveled = math.sqrt(
                    (self.odom.x - s_x) ** 2 + (self.odom.y - s_y) ** 2
                )
                remaining = abs(goal_dx - distance_traveled)
                logging.info(f"remaining advance GAP: {round(remaining,2)}")

                fb = 0
                if pp is not None and 4 in pp:
                    fb = 1
                elif pp is not None and 9 in pp:
                    fb = -1
                else:
                    logging.info("danger, pop 1 off queue")
                    self.pending_movements.get()
                    return

                if remaining > self.distance_tolerance:
                    if distance_traveled < goal_dx:  # keep advancing
                        logging.debug(f"keep moving. remaining:{remaining} ")
                        self.move(fb * 0.4, 0.0)
                    elif distance_traveled > goal_dx:  # you moved too far
                        logging.debug(
                            f"OVERSHOOT: move other way. remaining:{remaining} "
                        )
                        self.move(-1 * fb * 0.1, 0.0)
                else:
                    logging.info(
                        "advance is completed, gap is small enough, done, pop 1 off queue"
                    )
                    self.pending_movements.get()

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
            if not self.lidar.turn_left:
                logging.warning("Cannot turn left due to barrier")
                return False
            sharpness = min(self.lidar.turn_left)
            self.move(sharpness * 0.15, self.turn_speed)
        else:  # Turn right
            if not self.lidar.turn_right:
                logging.warning("Cannot turn right due to barrier")
                return False
            sharpness = 8 - max(self.lidar.turn_right)
            self.move(sharpness * 0.15, -self.turn_speed)
        return True
