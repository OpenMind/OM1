import logging
import math
import random
import time
from queue import Queue
from typing import List, Optional

import zenoh

from actions.base import ActionConfig, ActionConnector, MoveCommand
from actions.move_turtle.interface import MoveInput
from providers.odom_provider import OdomProvider
from providers.rplidar_provider import RPLidarProvider
from zenoh_idl import geometry_msgs, sensor_msgs


class MoveZenohConnector(ActionConnector[MoveInput]):

    def __init__(self, config: ActionConfig):

        super().__init__(config)

        self.angle_tolerance = 5.0
        self.distance_tolerance = 0.05  # m

        self.pending_movements: Queue[Optional[MoveCommand]] = Queue()

        self.hazard = None
        self.emergency = None

        self.session = None

        URID = getattr(self.config, "URID", None)

        if URID is None:
            logging.warning("Aborting TurtleBot4 Move system, no URID provided")
            return
        else:
            logging.info(f"TurtleBot4 Move system is using URID: {URID}")

        self.cmd_vel = f"{URID}/c3/cmd_vel"

        try:
            self.session = zenoh.open(zenoh.Config())
            logging.info(f"Zenoh move client opened {self.session}")
            logging.info(f"TurtleBot4 hazard listener starting with URID: {URID}")
            self.session.declare_subscriber(
                f"{URID}/c3/hazard_detection", self.listen_hazard
            )
        except Exception as e:
            logging.error(f"Error opening Zenoh client: {e}")

        self.lidar = RPLidarProvider()
        self.odom = OdomProvider(URID=URID, use_zenoh=True)

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
            and len(self.hazard.detections) > 0
        ):
            for haz in self.hazard.detections:
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

    def move(self, vx, vyaw):
        """
        generate movement commands
        """
        logging.debug("move: {} - {}".format(vx, vyaw))

        if self.session is None:
            logging.info("No open Zenoh session, returning")
            return

        logging.debug("Pub twist: {} - {}".format(vx, vyaw))
        t = geometry_msgs.Twist(
            linear=geometry_msgs.Vector3(x=float(vx), y=0.0, z=0.0),
            angular=geometry_msgs.Vector3(x=0.0, y=0.0, z=float(vyaw)),
        )
        self.session.put(self.cmd_vel, t.serialize())

    async def connect(self, output_interface: MoveInput) -> None:

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
            self.pending_movements.put(
                MoveCommand(dx=0.0, yaw=target_yaw, direction="turn")
            )
        elif output_interface.action == "turn right":
            # turn 90 Deg to the right (CW)
            target_yaw = self.odom.odom_yaw_m180_p180 + 30.0
            if target_yaw >= 180.0:
                target_yaw -= 360.0
            self.pending_movements.put(
                MoveCommand(dx=0.0, yaw=target_yaw, direction="turn")
            )
        elif output_interface.action == "move forwards":
            if advance_danger:
                return
            self.pending_movements.put(
                MoveCommand(
                    dx=0.5,
                    yaw=0.0,
                    direction="advance",
                    start_x=self.odom.x,
                    start_y=self.odom.y,
                )
            )
        elif output_interface.action == "move back":
            if retreat_danger:
                return
            self.pending_movements.put(
                MoveCommand(
                    dx=0.5,
                    yaw=0.0,
                    direction="retreat",
                    start_x=self.odom.x,
                    start_y=self.odom.y,
                )
            )
        elif output_interface.action == "stand still":
            logging.info(f"AI movement command: {output_interface.action}")
            # do nothing
        else:
            logging.info(f"AI movement command unknown: {output_interface.action}")

    def tick(self) -> None:

        time.sleep(0.1)

        logging.debug("Move tick")

        if self.odom.x == 0.0:
            # this value is never precisely zero except while
            # booting and waiting for data to arrive
            logging.info("Waiting for odom data")
            time.sleep(0.5)
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

            target = self.emergency
            logging.info(f"Emergency target: {target}")

            gap = self.odom.odom_yaw_m180_p180 - target
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
            direction = current_target.direction

            if "turn" in direction:
                gap = self.odom.odom_yaw_m180_p180 - goal_yaw
                if gap > 180.0:
                    gap -= 360.0
                elif gap < -180.0:
                    gap += 360.0
                logging.info(f"remaining turn GAP: {round(gap,2)}")
                if abs(gap) > 10.0:
                    logging.debug("gap is big, using large displacements")
                    if gap > 0:
                        self.move(0.0, 0.5)
                    elif gap < 0:
                        self.move(0.0, -0.5)
                elif abs(gap) > self.angle_tolerance and abs(gap) <= 10.0:
                    logging.debug("gap is getting smaller, using smaller steps")
                    if gap > 0:
                        self.move(0.0, 0.2)
                    elif gap < 0:
                        self.move(0.0, -0.2)
                elif abs(gap) <= self.angle_tolerance:
                    logging.info(
                        "turn is completed, gap is small enough, done, pop 1 off queue"
                    )
                    self.pending_movements.get()
            else:
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
                if "advance" in direction and 4 in pp:
                    fb = 1
                elif "retreat" in direction and 9 in pp:
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
