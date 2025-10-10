#!/usr/bin/env python3
"""
Light-weight odometry reader (Unitree DDS).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from inputs.base import SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

# ── Cyclone DDS ────────────────────────────────────────────────────────────
from providers.odom_provider import OdomProvider

# ─── constants ─────────────────────────────────────────────────────────────
R_EARTH = 6_371_000.0  # mean Earth radius (m)


# ── buffer helper ─────────────────────────────────────────────────────────
@dataclass
class Message:
    """
    Container for timestamped messages.

    Parameters
    ----------
    timestamp : float
        Unix timestamp of the message
    text : str
        Content of the message
    """

    timestamp: float
    text: str


# ── main reader class ──────────────────────────────────────────────────────
class GPSOdomReader(FuserInput[str]):
    """
    Maintains global pose (lat, lon, yaw) from Unitree Sport-mode state.
    """

    # ── init ─────────────────────────────────────────────────────────────
    def __init__(self, config: SensorConfig = SensorConfig()):
        super().__init__(config)

        # --- origin (deg) ------------------------------------------------
        self.lat0: float | None = getattr(config, "origin_lat", None)
        self.lon0: float | None = getattr(config, "origin_lon", None)
        yaw0_deg = getattr(config, "origin_yaw_deg", None)
        if self.lat0 is None or self.lon0 is None or yaw0_deg is None:
            logging.error(
                "GPSOdomReader: origin_lat, origin_lon, and origin_yaw_deg must be set in the config."
            )
            raise ValueError("Missing origin coordinates or yaw in config.")
        self._yaw_offset = math.radians(yaw0_deg) if yaw0_deg is not None else 0.0

        # --- current pose -----------------------------------------------
        self.pose_x = 0.0  # metres East  of origin
        self.pose_y = 0.0  # metres North of origin
        self.pose_yaw = 0.0  # rad

        # --- I/O + buffer ------------------------------------------------
        self.io_provider = IOProvider()
        self.buf: list[Message] = []
        self.descriptor_for_LLM = "Latitude, Longitude, and Yaw"

        unitree_ethernet: str | None = getattr(config, "unitree_ethernet", None)
        self.odom = OdomProvider(channel=unitree_ethernet)
        logging.info(f"Mapper Odom Provider: {self.odom}")

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _xy_to_latlon(self, x: float, y: float):
        φ0, λ0 = map(math.radians, (self.lat0, self.lon0))  # type: ignore
        φ = φ0 + y / R_EARTH
        λ = λ0 + x / (R_EARTH * math.cos(φ0))
        return map(math.degrees, (φ, λ))

    # ── pose update step ────────────────────────────────────────────────
    async def _update_pose(self):
        o = self.odom
        logging.debug(f"Odom data: {o}")
        self.pose_x = self.odom.x
        self.pose_y = self.odom.y
        yaw_world = math.radians(self.odom.odom_yaw_m180_p180)
        self.pose_yaw = self._wrap_angle(yaw_world + self._yaw_offset)

        # publish through IOProvider
        lat, lon = self._xy_to_latlon(self.pose_x, self.pose_y)
        self.io_provider.add_dynamic_variable("latitude", lat)
        self.io_provider.add_dynamic_variable("longitude", lon)
        self.io_provider.add_dynamic_variable("yaw_deg", math.degrees(self.pose_yaw))

    # ── polling loop (FuserInput interface) ─────────────────────────────
    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.5)
        await self._update_pose()
        return None

    # ---------------------------------------------------------------------
    # ChatGPT-bridge helpers — minimal but functional
    # ---------------------------------------------------------------------
    async def raw_to_text(self, raw_input: str):
        """
        Allow external callers (e.g. GUI, CLI) to push arbitrary text into
        the same IO/logging path that serial parsing used to feed.
        """
        if not raw_input:
            return
        now = time.time()
        text = raw_input.strip()
        self.buf.append(Message(now, text))
        self.io_provider.add_input(self.__class__.__name__, text, now)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Return the newest buffered message in the canonical // START … // END
        block that the LLM side expects, then clear the buffer.
        """
        if not self.buf:
            return None
        m = self.buf[-1]
        self.buf.clear()
        self.io_provider.add_input(self.__class__.__name__, m.text, m.timestamp)
        return f"""
{self.descriptor_for_LLM} INPUT
// START
{m.text}
// END
"""
