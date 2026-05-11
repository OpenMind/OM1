import asyncio
import json
import logging
import math
import random
import time
from queue import Queue
from typing import List, Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector, MoveCommand
from actions.move_k1_autonomy.interface import MoveInput
from providers.k1_odom_provider import K1OdomProvider, RobotState
from providers.simple_paths_provider import SimplePathsProvider
from zenoh_msgs import (
    BoosterApiReqMsg,
    RpcServiceRequest,
    RpcServiceResponse,
    open_zenoh_session,
)


class MoveBoosterZenohConfig(ActionConfig):
    """
    Configuration for Booster Zenoh connector.
    """

    odom_topic: str = Field(
        default="odometer_state",
        description="Zenoh topic for odometry data.",
    )
    rpc_service_name: str = Field(
        default="booster_rpc_service",
        description="Zenoh key for the Booster ROS2 RPC service (e.g. booster_rpc_service).",
    )

    # Backward-compat: older configs used cmd_vel_topic for topic-based control.
    # If provided, we treat it as the RPC service name.
    cmd_vel_topic: Optional[str] = Field(
        default=None,
        description="DEPRECATED. Previously used for remote_controller_state topic; now interpreted as rpc_service_name.",
    )

    allow_move_without_odom: bool = Field(
        default=False,
        description="TESTING ONLY. If true, bypass odom/body-attitude gating and send movement RPC commands even when odom is missing.",
    )


class MoveBoosterZenohConnector(ActionConnector[MoveBoosterZenohConfig, MoveInput]):
    """
    Zenoh connector for the Move Booster autonomy action.
    Uses Zenoh to publish cmd_vel commands and receive odom data from Booster robot.
    """

    def __init__(self, config: MoveBoosterZenohConfig):
        """
        Initialize the Zenoh connector for Booster robot.

        Parameters
        ----------
        config : MoveBoosterZenohConfig
            The configuration for the action connector.
        """
        super().__init__(config)

        # Movement parameters
        self.move_speed = 0.1
        self.turn_speed = 0.5  # 0.35
        self.angle_tolerance = 5.0  # degrees
        self.distance_tolerance = 0.05  # meters
        self.pending_movements: Queue[Optional[MoveCommand]] = Queue()
        self.movement_attempts = 0
        self.movement_attempt_limit = 15
        self.gap_previous = 0
        self._consecutive_retreat_cmds = 0  # calculate whether 5 time retreat (move back), to prevent the odom inaccuracy causing continuous retreating

        self.session = None

        odom_topic = self.config.odom_topic
        self.rpc_service_name = self.config.rpc_service_name or self.config.cmd_vel_topic or "booster_rpc_service"

        try:
            self.session = open_zenoh_session()
            logging.info(f"Booster Zenoh move client opened {self.session}")
        except Exception as e:
            logging.error(f"Error opening Zenoh client for Booster: {e}")

        self.path_provider = SimplePathsProvider()
        self.odom = K1OdomProvider(topic=odom_topic)

        logging.info(f"Booster Autonomy Odom Provider: {self.odom}")
        logging.info(f"Booster Autonomy RPC service key: {self.rpc_service_name}")

    def _has_fresh_odom(self, max_age_s: float = 2.0) -> bool:
        # K1 odometry can legitimately be exactly (0.0, 0.0) while stationary,
        # so we must not use odom_x==0.0 as a readiness signal.
        ts = float(self.odom.position.get("odom_subscriber_ts", 0.0))
        if ts <= 0.0:
            return False
        return (time.time() - ts) <= max_age_s

    def _run_move_robot(self, vx: float, vy: float, vyaw: float) -> None:
        # Called from both sync code (tick thread) and async code (connect).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._move_robot(vx, vy, vyaw))
        else:
            loop.create_task(self._move_robot(vx, vy, vyaw))

    def _stop_robot(self) -> None:
        try:
            self._run_move_robot(0.0, 0.0, 0.0)
        except Exception as e:
            logging.debug(f"Stop robot failed: {e}")

    async def _move_robot(self, vx: float, vy: float, vyaw: float) -> None:
        """
        Send movement command via Zenoh RPC service booster_rpc_service.

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

        if not self.config.allow_move_without_odom:
            if self.odom.position["body_attitude"] != RobotState.STANDING:
                logging.info("Cannot move - robot is not standing")
                return

        try:
            API_MOVE = 2001
            # Create the inner request message
            inner_request = BoosterApiReqMsg(api_id=API_MOVE, body=json.dumps({"vx": vx, "vy": vy, "vyaw": vyaw}))
            # Wrap it in RpcServiceRequest
            request = RpcServiceRequest(msg=inner_request)
            # Serialize for Zenoh bridge
            serialized_request = request.serialize()
            service_name = self.rpc_service_name
            # TODO: Add support for cloud sim for the service call
            replies = self.session.get(  # type: ignore
                service_name,
                payload=serialized_request,
                timeout=5.0,
            )
            for reply in replies:
                if reply.ok:
                    try:
                        service_response = RpcServiceResponse.deserialize(reply.ok.payload.to_bytes())
                        logging.info(
                            f"RPC response status: {service_response.msg.status}, body: {service_response.msg.body}"
                        )
                    except Exception as e:
                        logging.error(f"Error deserializing response: {e}")
                else:
                    logging.error(f"Service error: {reply.err}")
        except Exception as e:
            logging.error(f"Service call failed: {e}")

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect to the output interface and process the AI movement command.

        Parameters
        ----------
        output_interface : MoveInput
            The output interface containing the AI movement command.
        """
        logging.info(f"Booster AI command.connect: {output_interface.action}")

        if self.odom.position["moving"]:
            logging.info("Disregard new AI movement command - robot is already moving")
            return

        if self.pending_movements.qsize() > 0:
            logging.info("Movement in progress: disregarding new AI command")
            return

        if not self._has_fresh_odom():
            if self.config.allow_move_without_odom:
                logging.warning("ODOM missing/stale but allow_move_without_odom=true; sending direct test command")

                action = output_interface.action
                if action == "move forwards":
                    self._run_move_robot(0.1, 0.0, 0.0)
                elif action == "move back":
                    self._run_move_robot(-0.1, 0.0, 0.0)
                elif action == "turn left":
                    self._run_move_robot(0.0, 0.0, 0.15)
                elif action == "turn right":
                    self._run_move_robot(0.0, 0.0, -0.15)
                elif action == "stand still":
                    self._stop_robot()
                else:
                    logging.info(f"AI movement command unknown: {action}")
                return

            logging.info("Waiting for fresh odom data")
            return

        # Process movement commands with lidar safety checks
        movement_map = {
            "turn left": self._process_turn_left,
            "turn right": self._process_turn_right,
            "move forwards": self._process_move_forward,
            "move back": self._process_move_back,
            "stand still": self._stop_robot,
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
        self._stop_robot()
        self.movement_attempts = 0
        self._consecutive_retreat_cmds = 0
        if not self.pending_movements.empty():
            self.pending_movements.get()

    def tick(self) -> None:
        """
        Process the AI motion tick.
        """
        logging.debug("Booster AI Motion Tick")

        if self.odom is None:
            logging.info("Waiting for odom data = self.odom is None")
            self.sleep(0.5)
            return

        if not self._has_fresh_odom():
            logging.info("Waiting for fresh odom data")
            self.sleep(0.5)
            return

        if self.odom.position["body_attitude"] != RobotState.STANDING:
            logging.info("Cannot move - robot is not standing")
            self.sleep(0.5)
            return

        # if we got to this point, we have good data and we are able to
        # safely proceed
        with self.pending_movements.mutex:
            target: List[MoveCommand] = list(self.pending_movements.queue)

        if len(target) > 0:

            current_target = target[0]

            logging.info(f"Target: {current_target} current yaw: {self.odom.position['odom_yaw_m180_p180']}")

            if self.movement_attempts > self.movement_attempt_limit:
                # abort - we are not converging
                self.clean_abort()
                logging.info(f"TIMEOUT - not converging after {self.movement_attempt_limit} attempts - StopMove()")
                return

            goal_dx = current_target.dx
            goal_yaw = current_target.yaw

            # Phase 1: Turn to face the target direction
            if not current_target.turn_complete:
                # Turning resets consecutive retreat accounting.
                self._consecutive_retreat_cmds = 0
                gap = self._calculate_angle_gap(-1 * self.odom.position["odom_yaw_m180_p180"], goal_yaw)
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
                        self._run_move_robot(0, 0, 0.2)
                    elif gap < 0:
                        self._run_move_robot(0, 0, -0.2)
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

                # If we've been retreating for several ticks and odom/progress is unreliable,
                # stop retreating and decide what to do next based only on obstacle safety.
                if goal_dx < 0 and self._consecutive_retreat_cmds >= 5:
                    logging.warning(
                        "Retreat attempted 5+ times; aborting retreat and checking if front is clear to move forward"
                    )
                    self.clean_abort()
                    if self._enqueue_forward_if_front_clear(dx=0.1):
                        self.movement_attempts = 0
                        self.gap_previous = 0
                    return
                if goal_dx >= 0:
                    # Any non-retreat target resets consecutive retreat accounting.
                    self._consecutive_retreat_cmds = 0

                s_x = current_target.start_x
                s_y = current_target.start_y
                speed = current_target.speed

                distance_traveled = math.sqrt(
                    (self.odom.position["odom_x"] - s_x) ** 2 + (self.odom.position["odom_y"] - s_y) ** 2
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
                        self._run_move_robot(fb * speed, 0.0, 0.0)
                        if fb < 0:
                            self._consecutive_retreat_cmds += 1
                        else:
                            self._consecutive_retreat_cmds = 0
                    elif distance_traveled > abs(goal_dx):
                        logging.debug(f"Phase 2 - OVERSHOOT: move other way. Remaining: {gap}m")
                        self._run_move_robot(-1 * fb * 0.15, 0.0, 0.0)
                        # Overshoot correction is opposite direction; reset retreat accounting.
                        self._consecutive_retreat_cmds = 0
                else:
                    logging.info("Phase 2 - Movement completed normally, processing next AI command")
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

        target_yaw = self._normalize_angle(-1 * self.odom.position["odom_yaw_m180_p180"] + path_angle)
        self.pending_movements.put(
            MoveCommand(
                dx=0.0,
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

        target_yaw = self._normalize_angle(-1 * self.odom.position["odom_yaw_m180_p180"] + path_angle)
        self.pending_movements.put(
            MoveCommand(
                dx=0.0,
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

        target_yaw = self._normalize_angle(-1 * self.odom.position["odom_yaw_m180_p180"] + path_angle)
        self.pending_movements.put(
            MoveCommand(
                dx=0.1,
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
                dx=-0.15,
                yaw=0.0,
                start_x=round(self.odom.position["odom_x"], 2),
                start_y=round(self.odom.position["odom_y"], 2),
                turn_complete=True,
                speed=0.1,
            )
        )

    def _enqueue_forward_if_front_clear(self, dx: float = 0.1) -> bool:
        # "Front clear" == straight path (index 4) is available.
        if 4 not in (self.path_provider.advance or []):
            logging.warning("Front not clear (path 4 not safe); staying stopped")
            return False

        target_yaw = self._normalize_angle(-1 * self.odom.position["odom_yaw_m180_p180"])
        self.pending_movements.put(
            MoveCommand(
                dx=dx,
                yaw=target_yaw,
                start_x=round(self.odom.position["odom_x"], 2),
                start_y=round(self.odom.position["odom_y"], 2),
                turn_complete=True,
                speed=self.move_speed,
            )
        )
        return True

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
            self._run_move_robot(sharpness * 0.15, 0, self.turn_speed)
        else:  # Turn right
            if not self.path_provider.turn_right:
                logging.warning("Cannot turn right due to barrier")
                return False
            sharpness = 8 - max(self.path_provider.turn_right)
            self._run_move_robot(sharpness * 0.15, 0, -self.turn_speed)
        return True
