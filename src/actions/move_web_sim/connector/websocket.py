import logging
import math
import time
from queue import Queue
from typing import Optional

import requests
from pydantic import Field

from actions.base import ActionConfig, ActionConnector, MoveCommand
from actions.move_web_sim.interface import MoveInput


class MoveWebSimConfig(ActionConfig):
    """Configuration for WebSim connector."""

    simulator_url: str = Field(
        default="http://localhost:8001",
        description="Base URL for the three.js simulator HTTP API.",
    )


class MoveWebSimConnector(ActionConnector[MoveWebSimConfig, MoveInput]):
    """HTTP connector for the Move WebSim action."""

    def __init__(self, config: MoveWebSimConfig):
        super().__init__(config)

        self.simulator_url = config.simulator_url

        self.turn_speed = 0.8
        self.angle_tolerance = 5.0
        self.distance_tolerance = 0.05
        self.pending_movements: Queue[Optional[MoveCommand]] = Queue()
        self.movement_attempts = 0
        self.movement_attempt_limit = 15
        self.gap_previous = 0

        self.robot_state = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "moving": False,
        }

        logging.info(f"WebSim connector initialized, simulator URL: {self.simulator_url}")

    def _send_command(self, command: dict):
        try:
            response = requests.post(
                f"{self.simulator_url}/api/command",
                json=command,
                timeout=0.5,
            )
            if response.status_code == 200:
                data = response.json()
                if "robot_state" in data:
                    self.robot_state.update(data["robot_state"])
        except requests.exceptions.RequestException:
            logging.debug(f"Could not send command to simulator: {command}")
        except Exception as e:
            logging.debug(f"Error sending command: {e}")

    def _send_command_sync(self, command: dict):
        self._send_command(command)

    async def connect(self, output_interface: MoveInput) -> None:
        """Connect to the output interface and process the AI movement command."""
        logging.info(f"AI command.connect: {output_interface.action}")

        if self.pending_movements.qsize() > 0:
            logging.info("Movement in progress: disregarding new AI command")
            return

        if self.robot_state["moving"]:
            logging.info("Robot is already moving, disregarding new AI command")
            return

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

    def _process_turn_left(self):
        target_yaw = self._normalize_angle(self.robot_state["yaw"] - 90.0)
        self.pending_movements.put(
            MoveCommand(
                dx=0.0,
                yaw=target_yaw,
                start_x=self.robot_state["x"],
                start_y=self.robot_state["y"],
                turn_complete=False,
            )
        )
        self._send_command_sync({"type": "turn_left", "target_yaw": target_yaw})

    def _process_turn_right(self):
        target_yaw = self._normalize_angle(self.robot_state["yaw"] + 90.0)
        self.pending_movements.put(
            MoveCommand(
                dx=0.0,
                yaw=target_yaw,
                start_x=self.robot_state["x"],
                start_y=self.robot_state["y"],
                turn_complete=False,
            )
        )
        self._send_command_sync({"type": "turn_right", "target_yaw": target_yaw})

    def _process_move_forward(self):
        self.pending_movements.put(
            MoveCommand(
                dx=0.5,
                yaw=self.robot_state["yaw"],
                start_x=self.robot_state["x"],
                start_y=self.robot_state["y"],
                turn_complete=True,
            )
        )
        self._send_command_sync({"type": "move_forward", "distance": 0.5})

    def _process_move_back(self):
        self.pending_movements.put(
            MoveCommand(
                dx=-0.5,
                yaw=self.robot_state["yaw"],
                start_x=self.robot_state["x"],
                start_y=self.robot_state["y"],
                turn_complete=True,
                speed=0.3,
            )
        )
        self._send_command_sync({"type": "move_back", "distance": 0.5})

    def _normalize_angle(self, angle: float) -> float:
        while angle < -180:
            angle += 360.0
        while angle > 180:
            angle -= 360.0
        return angle

    def _calculate_angle_gap(self, current: float, target: float) -> float:
        gap = current - target
        if gap > 180.0:
            gap -= 360.0
        elif gap < -180.0:
            gap += 360.0
        return round(gap, 2)

    def clean_abort(self) -> None:
        """Cleanly abort current movement and reset state."""
        self.movement_attempts = 0
        self.robot_state["moving"] = False
        if not self.pending_movements.empty():
            self.pending_movements.get()
        self._send_command_sync({"type": "stop"})

    def tick(self) -> None:
        """Process the AI motion tick."""
        time.sleep(0.1)

        target: list[MoveCommand] = list(self.pending_movements.queue)

        if len(target) > 0:
            current_target = target[0]

            goal_dx = current_target.dx
            goal_yaw = current_target.yaw

            if not current_target.turn_complete:
                gap = self._calculate_angle_gap(self.robot_state["yaw"], goal_yaw)
                logging.debug(f"Turning remaining GAP: {gap}DEG")

                if abs(gap) > self.angle_tolerance:
                    self.movement_attempts += 1
                    self.robot_state["moving"] = True
                    turn_speed = 0.2 if abs(gap) > 10.0 else 0.1
                    if gap > 0:
                        self.robot_state["yaw"] = self._normalize_angle(
                            self.robot_state["yaw"] - turn_speed * 10
                        )
                        self._send_command_sync(
                            {"type": "rotate", "speed": -turn_speed}
                        )
                    else:
                        self.robot_state["yaw"] = self._normalize_angle(
                            self.robot_state["yaw"] + turn_speed * 10
                        )
                        self._send_command_sync({"type": "rotate", "speed": turn_speed})
                else:
                    logging.info("Turn completed, starting movement")
                    current_target.turn_complete = True
                    self.robot_state["yaw"] = goal_yaw

            else:
                if goal_dx == 0:
                    logging.info("No movement required, processing next AI command")
                    self.clean_abort()
                    return

                s_x = current_target.start_x
                s_y = current_target.start_y
                speed = current_target.speed

                distance_traveled = math.sqrt(
                    (self.robot_state["x"] - s_x) ** 2
                    + (self.robot_state["y"] - s_y) ** 2
                )
                gap = round(abs(goal_dx) - distance_traveled, 2)

                if gap > self.distance_tolerance:
                    self.movement_attempts += 1
                    self.robot_state["moving"] = True
                    yaw_rad = math.radians(self.robot_state["yaw"])
                    move_distance = speed * 0.1
                    self.robot_state["x"] += move_distance * math.cos(yaw_rad)
                    self.robot_state["y"] += move_distance * math.sin(yaw_rad)

                    if goal_dx > 0:
                        self._send_command_sync({"type": "move", "speed": speed})
                    else:
                        self._send_command_sync({"type": "move", "speed": -speed})
                else:
                    logging.info("Movement completed, processing next AI command")
                    self.clean_abort()

            if self.movement_attempts > self.movement_attempt_limit:
                logging.warning("Movement timeout, aborting")
                self.clean_abort()
        else:
            self.robot_state["moving"] = False

