"""Unitree Go2 odometry input plugin (Zenoh transport).

Reads pose data from a Zenoh keyexpression (default ``utlidar/robot_pose``)
via ``open_zenoh_session()``. Surface-compatible with ``UnitreeGo2Odom``.

    agent_inputs: [
        { type: "UnitreeGo2OdomZenoh" },
    ]
"""

from __future__ import annotations

import asyncio
import logging
import time
from queue import Empty, Queue
from typing import List, Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.odom_provider_base import RobotState
from providers.unitree_go2_odom_zenoh_provider import UnitreeGo2OdomZenohProvider


class UnitreeGo2OdomZenohConfig(SensorConfig):
    """Configuration for ``UnitreeGo2OdomZenoh``.

    Parameters
    ----------
    topic : str
        Zenoh keyexpression to subscribe to.
    message_schema : str
        ``geometry_msgs/msg/PoseStamped`` or ``nav_msgs/msg/Odometry``.
    """

    topic: str = Field(
        default="utlidar/robot_pose",
        description="Zenoh key for Go2 robot_pose / odom.",
    )
    message_schema: str = Field(
        default="geometry_msgs/msg/PoseStamped",
        description="Message schema for the odom topic.",
    )


class UnitreeGo2OdomZenoh(FuserInput[UnitreeGo2OdomZenohConfig, Optional[dict]]):
    """Zenoh-routed Go2 odometry input."""

    def __init__(self, config: UnitreeGo2OdomZenohConfig):
        super().__init__(config)
        self.io_provider = IOProvider()
        self.messages: List[Message] = []
        self.message_buffer: Queue[str] = Queue()

        logging.info(f"Config: {self.config}")

        self.odom = UnitreeGo2OdomZenohProvider(
            topic=self.config.topic,
            schema=self.config.message_schema,
        )
        self.descriptor_for_LLM = "Information about your location and body pose, to help plan your movements."

    async def _poll(self) -> Optional[dict]:
        await asyncio.sleep(0.1)
        try:
            return self.odom.position
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[dict]) -> Optional[Message]:
        if raw_input is None:
            return None

        moving = raw_input["moving"]
        attitude = raw_input["body_attitude"]

        if attitude is RobotState.SITTING:
            res = "You are sitting down - do not generate new movement commands. "
        elif moving:
            res = "You are moving - do not generate new movement commands. "
        else:
            res = "You are standing still - you can move if you want to. "

        return Message(timestamp=time.time(), message=res)

    async def raw_to_text(self, raw_input: Optional[dict]):
        """Replace the buffered odom message with the latest decoded sample."""
        msg = await self._raw_to_text(raw_input)
        if msg is not None:
            if len(self.messages) == 0:
                self.messages.append(msg)
            else:
                self.messages[-1] = msg

    def formatted_latest_buffer(self) -> Optional[str]:
        """Return and clear the most recent formatted odom message."""
        if not self.messages:
            return None
        latest = self.messages[-1]
        result = f"\nINPUT: {self.descriptor_for_LLM}\n// START\n{latest.message}\n// END\n"
        self.io_provider.add_input(self.__class__.__name__, latest.message, latest.timestamp)
        self.messages = []
        return result
