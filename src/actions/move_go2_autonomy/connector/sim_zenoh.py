import json
import logging
import math
import random
import threading
import time
from queue import Queue
from typing import List, Optional

import zenoh
from pydantic import Field
from zenoh import ZBytes

from actions.base import ActionConfig, ActionConnector, MoveCommand
from actions.move_go2_autonomy.interface import MoveInput
from providers.simple_paths_provider import SimplePathsProvider
from zenoh_msgs import (
    AIStatusRequest,
    AIStatusResponse,
    PoseStamped,
    String,
    open_zenoh_session,
    prepare_header,
    UnitreeRequest,
    UnitreeRequestHeader,
    UnitreeRequestIdentity,
)


SPORT_REQUEST_TOPIC = "api/sport/request"
SPORT_API_ID_MOVE = 1008
SPORT_API_ID_BALANCESTAND = 1002
SPORT_API_ID_STOPMOVE = 1003

ROBOT_POSE_TOPIC = "utlidar/robot_pose"


class MoveGo2SimZenohConfig(ActionConfig):
    """Configuration for Go2 simulation movement over Zenoh."""

    mode: Optional[str] = Field(default=None, description="Optional operating mode name.")


class MoveGo2SimZenohConnector(ActionConnector[MoveGo2SimZenohConfig, MoveInput]):
    """
    Connector for Go2 simulation movement using Zenoh, matching the behavior
    of MoveUnitreeOMPathSDKConnector with odometry-based tick loop movement.
    """

    def __init__(self, config: MoveGo2SimZenohConfig):
        super().__init__(config)
        self.session: Optional[zenoh.Session] = None

        # Movement parameters (matching unitree_om_path_sdk)
        self.move_speed = 0.5
        self.turn_speed = 0.8
        self.angle_tolerance = 5.0  # degrees
        self.distance_tolerance = 0.05  # meters
        self.pending_movements: Queue[Optional[MoveCommand]] = Queue()
        self.movement_attempts = 0
        self.movement_attempt_limit = 15
        self.gap_previous = 0

        self.path_provider = SimplePathsProvider()

        # Odometry state
        self._odom_lock = threading.Lock()
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw_m180_p180 = 0.0
        self._body_height_cm = 0
        self._odom_ready = False

        # AI control status
        self.ai_control_enabled = True
        self.ai_status_request = "om/ai/request"
        self.ai_status_response = "om/ai/response"

        # Mode
        self.mode = self.config.mode

        try:
            self.session = open_zenoh_session()
            # Subscribe to robot pose for odometry
            self.session.declare_subscriber(ROBOT_POSE_TOPIC, self._on_robot_pose)
            # Subscribe to AI control status
            self.session.declare_subscriber(self.ai_status_request, self._zenoh_ai_status_request)
            self._zenoh_ai_status_response_pub = self.session.declare_publisher(self.ai_status_response)
            logging.info("MoveGo2SimZenohConnector: Zenoh session opened, subscribed to odom")
        except Exception as e:
            logging.error(f"MoveGo2SimZenohConnector: failed to open Zenoh session: {e}")

    def _on_robot_pose(self, sample: zenoh.Sample) -> None:
        """Process incoming PoseStamped from utlidar/robot_pose."""
        try:
            pose_stamped = PoseStamped.deserialize(sample.payload.to_bytes())
            pos = pose_stamped.pose.position
            ori = pose_stamped.pose.orientation

            # Quaternion to yaw (euler Z)
            siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
            cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw_rad)

            body_height_cm = round(pos.z * 100.0)

            with self._odom_lock:
                self._odom_x = pos.x
                self._odom_y = pos.y
                self._odom_yaw_m180_p180 = yaw_deg
                self._body_height_cm = body_height_cm
                self._odom_ready = True
        except Exception as e:
            logging.debug(f"MoveGo2SimZenohConnector: error parsing pose: {e}")

    @property
    def odom_x(self) -> float:
        with self._odom_lock:
            return self._odom_x

    @property
    def odom_y(self) -> float:
        with self._odom_lock:
            return self._odom_y

    @property
    def odom_yaw(self) -> float:
        with self._odom_lock:
            return self._odom_yaw_m180_p180

    @property
    def body_height_cm(self) -> int:
        with self._odom_lock:
            return self._body_height_cm

    @property
    def is_standing(self) -> bool:
        return self.body_height_cm > 24

    @property
    def odom_ready(self) -> bool:
        with self._odom_lock:
            return self._odom_ready

    def _publish_request(self, api_id: int, parameter: str = "") -> None:
        if self.session is None:
            return
        identity = UnitreeRequestIdentity(id=0, api_id=api_id)
        header = UnitreeRequestHeader(identity=identity)
        request = UnitreeRequest(header=header, parameter=parameter)
        self.session.put(SPORT_REQUEST_TOPIC, ZBytes(request.serialize()))

    def _send_move(self, vx: float, vy: float = 0.0, vturn: float = 0.0) -> None:
        logging.info(f"_send_move: vx={vx}, vy={vy}, vturn={vturn}")
        self._publish_request(
            SPORT_API_ID_MOVE,
            json.dumps({"x": float(vx), "y": float(vy), "z": float(vturn)}),
        )

    def _send_stop(self) -> None:
        self._publish_request(SPORT_API_ID_STOPMOVE)

    def _send_balance_stand(self) -> None:
        self._publish_request(SPORT_API_ID_BALANCESTAND)

    def clean_abort(self) -> None:
        self.movement_attempts = 0
        if not self.pending_movements.empty():
            self.pending_movements.get()

    async def connect(self, output_interface: MoveInput) -> None:
        action = output_interface.action
        logging.info(f"MoveGo2SimZenohConnector: AI command.connect: {action}")

        if not self.ai_control_enabled:
            logging.info("AI Control is disabled - disregarding AI command")
            return

        if self.session is None:
            logging.warning("MoveGo2SimZenohConnector: no Zenoh session")
            return

        if self.pending_movements.qsize() > 0:
            logging.info("Movement in progress: disregarding new AI command")
            return

        if not self.odom_ready:
            logging.info("Waiting for odom data")
            return

        if self.odom_x == 0.0:
            logging.info("Waiting for location data")
            return

        movement_map = {
            "turn left": self._process_turn_left,
            "turn right": self._process_turn_right,
            "move forwards": self._process_move_forward,
            "move back": self._process_move_back,
            "stand still": lambda: logging.info("AI movement command: stand still"),
        }

        handler = movement_map.get(action)
        if handler:
            handler()
        else:
            logging.info(f"AI movement command unknown: {action}")

    def tick(self) -> None:
        logging.debug("AI Motion Tick")

        if not self.odom_ready:
            logging.info("Waiting for odom data")
            self.sleep(0.5)
            return

        if self.odom_x == 0.0:
            logging.info("Waiting for odom data, x == 0.0")
            self.sleep(0.5)
            return

        if not self.is_standing:
            logging.info("Cannot move - robot is not standing")
            self.sleep(0.5)
            return

        target: List[MoveCommand] = list(self.pending_movements.queue)

        if len(target) > 0:
            current_target = target[0]

            logging.info(f"Target: {current_target} current yaw: {self.odom_yaw}")

            if self.movement_attempts > self.movement_attempt_limit:
                self.clean_abort()
                logging.info(f"TIMEOUT - not converging after {self.movement_attempt_limit} attempts")
                return

            goal_dx = current_target.dx
            goal_yaw = current_target.yaw

            # Phase 1: Turn to face the target direction
            if not current_target.turn_complete:
                gap = self._calculate_angle_gap(-1 * self.odom_yaw, goal_yaw)
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
                    if gap > 0:
                        self._send_move(0, 0, 0.2)
                    elif gap < 0:
                        self._send_move(0, 0, -0.2)
                elif abs(gap) <= self.angle_tolerance:
                    logging.info("Phase 1 - Turn completed, starting movement")
                    current_target.turn_complete = True
                    self.gap_previous = 0

            else:
                # Phase 2: Move towards the target position
                if goal_dx == 0:
                    logging.info("No movement required, processing next AI command")
                    self.clean_abort()
                    return

                s_x = current_target.start_x
                s_y = current_target.start_y
                speed = current_target.speed

                distance_traveled = math.sqrt(
                    (self.odom_x - s_x) ** 2 + (self.odom_y - s_y) ** 2
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
                        logging.info(f"Phase 2 - Keep moving. Remaining: {gap}m")
                        self._send_move(fb * speed, 0.0, 0.0)
                    elif distance_traveled > abs(goal_dx):
                        logging.debug(f"Phase 2 - OVERSHOOT: move other way. Remaining: {gap}m")
                        self._send_move(-1 * fb * 0.2, 0.0, 0.0)
                else:
                    logging.info("Phase 2 - Movement completed normally, processing next AI command")
                    self.clean_abort()

        self.sleep(0.1)

    def _process_turn_left(self):
        if not self.path_provider.turn_left:
            logging.warning("Cannot turn left due to barrier")
            return

        path = random.choice(self.path_provider.turn_left)
        path_angle = self.path_provider.path_angles[path]

        target_yaw = self._normalize_angle(-1 * self.odom_yaw + path_angle)
        self.pending_movements.put(
            MoveCommand(
                dx=0.5,
                yaw=round(target_yaw, 2),
                start_x=round(self.odom_x, 2),
                start_y=round(self.odom_y, 2),
                turn_complete=False,
            )
        )

    def _process_turn_right(self):
        if not self.path_provider.turn_right:
            logging.warning("Cannot turn right due to barrier")
            return

        path = random.choice(self.path_provider.turn_right)
        path_angle = self.path_provider.path_angles[path]

        target_yaw = self._normalize_angle(-1 * self.odom_yaw + path_angle)
        self.pending_movements.put(
            MoveCommand(
                dx=0.5,
                yaw=round(target_yaw, 2),
                start_x=round(self.odom_x, 2),
                start_y=round(self.odom_y, 2),
                turn_complete=False,
            )
        )

    def _process_move_forward(self):
        if not self.path_provider.advance:
            logging.warning("Cannot advance due to barrier")
            return

        path = random.choice(self.path_provider.advance)
        path_angle = self.path_provider.path_angles[path]

        target_yaw = self._normalize_angle(-1 * self.odom_yaw + path_angle)
        self.pending_movements.put(
            MoveCommand(
                dx=0.5,
                yaw=target_yaw,
                start_x=round(self.odom_x, 2),
                start_y=round(self.odom_y, 2),
                turn_complete=path_angle == 0,
            )
        )

    def _process_move_back(self):
        if not self.path_provider.retreat:
            logging.warning("Cannot retreat due to barrier")
            return

        self.pending_movements.put(
            MoveCommand(
                dx=-0.5,
                yaw=0.0,
                start_x=round(self.odom_x, 2),
                start_y=round(self.odom_y, 2),
                turn_complete=True,
                speed=0.2,
            )
        )

    def _normalize_angle(self, angle: float) -> float:
        if angle < -180:
            angle += 360.0
        elif angle > 180:
            angle -= 360.0
        return angle

    def _calculate_angle_gap(self, current: float, target: float) -> float:
        gap = current - target
        if gap > 180.0:
            gap -= 360.0
        elif gap < -180.0:
            gap += 360.0
        return round(gap, 2)

    def _execute_turn(self, gap: float) -> bool:
        if gap > 0:  # Turn left
            if not self.path_provider.turn_left:
                logging.warning("Cannot turn left due to barrier")
                return False
            sharpness = min(self.path_provider.turn_left)
            self._send_move(sharpness * 0.15, 0, self.turn_speed)
        else:  # Turn right
            if not self.path_provider.turn_right:
                logging.warning("Cannot turn right due to barrier")
                return False
            sharpness = 8 - max(self.path_provider.turn_right)
            self._send_move(sharpness * 0.15, 0, -self.turn_speed)
        return True

    def _zenoh_ai_status_request(self, data: zenoh.Sample):
        ai_control_status = AIStatusRequest.deserialize(data.payload.to_bytes())
        logging.info(f"Received AI Control Status message: {ai_control_status}")

        code = ai_control_status.code
        request_id = ai_control_status.request_id

        if code == 2:
            ai_status_response = AIStatusResponse(
                header=prepare_header(ai_control_status.header.frame_id),
                request_id=request_id,
                code=1 if self.ai_control_enabled else 0,
                status=String(data=("AI Control Enabled" if self.ai_control_enabled else "AI Control Disabled")),
            )
            return self._zenoh_ai_status_response_pub.put(ai_status_response.serialize())

        if code == 1:
            self.ai_control_enabled = True
            logging.info("AI Control Enabled")
            ai_status_response = AIStatusResponse(
                header=prepare_header(ai_control_status.header.frame_id),
                request_id=request_id,
                code=1,
                status=String(data="AI Control Enabled"),
            )
            return self._zenoh_ai_status_response_pub.put(ai_status_response.serialize())

        if code == 0:
            self.ai_control_enabled = False
            logging.info("AI Control Disabled")
            ai_status_response = AIStatusResponse(
                header=prepare_header(ai_control_status.header.frame_id),
                request_id=request_id,
                code=0,
                status=String(data="AI Control Disabled"),
            )
            return self._zenoh_ai_status_response_pub.put(ai_status_response.serialize())

    def stop(self) -> None:
        self._send_stop()
        if self.session is not None:
            self.session.close()
            self.session = None
            logging.info("MoveGo2SimZenohConnector: Zenoh session closed")