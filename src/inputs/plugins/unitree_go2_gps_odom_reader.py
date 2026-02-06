import asyncio
import logging
import math
import time
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider
from providers.unitree_go2_odom_provider import UnitreeGo2OdomProvider

R_EARTH = 6_371_000.0  # mean Earth radius (m)


class UnitreeGo2GPSOdomReaderConfig(SensorConfig):
    """
    Configuration for GPS and Unitree Go2 Odom Reader Sensor.

    Parameters
    ----------
    origin_lat : Optional[float]
        Origin Latitude.
    origin_lon : Optional[float]
        Origin Longitude.
    origin_yaw_deg : Optional[float]
        Origin Yaw Degrees.
    unitree_ethernet : Optional[str]
        Unitree Ethernet Interface.
    """

    origin_lat: Optional[float] = Field(default=None, description="Origin Latitude")
    origin_lon: Optional[float] = Field(default=None, description="Origin Longitude")
    origin_yaw_deg: Optional[float] = Field(
        default=None, description="Origin Yaw Degrees"
    )
    unitree_ethernet: Optional[str] = Field(
        default=None, description="Unitree Ethernet Interface"
    )


class UnitreeGo2GPSOdomReader(FuserInput[UnitreeGo2GPSOdomReaderConfig, Optional[str]]):
    """
    GPS Odom Reader input handler for reading Unitree Go2 odometry data and converting it to GPS coordinates.
    """

    def __init__(self, config: UnitreeGo2GPSOdomReaderConfig):
        """
        Initialize the GPSOdomReader input handler.

        Sets up the required providers and buffers for handling GPS/odometry data.
        Initializes variables for origin coordinates and yaw offset.

        Parameters
        ----------
        config : GPSOdomReaderConfig
            Configuration for the sensor input, including origin coordinates and yaw.
        """
        super().__init__(config)

        self.lat0 = self.config.origin_lat
        self.lon0 = self.config.origin_lon
        yaw0_deg = self.config.origin_yaw_deg
        if self.lat0 is None or self.lon0 is None or yaw0_deg is None:
            logging.error(
                "GPSOdomReader: origin_lat, origin_lon, and origin_yaw_deg must be set in the config."
            )
            raise ValueError("Missing origin coordinates or yaw in config.")
        self._yaw_offset = math.radians(yaw0_deg) if yaw0_deg is not None else 0.0

        self.pose_x = 0.0  # metres East  of origin
        self.pose_y = 0.0  # metres North of origin
        self.pose_yaw = 0.0  # rad

        self.io_provider = IOProvider()
        self.buf: list[Message] = []
        self.descriptor_for_LLM = "Latitude, Longitude, and Yaw"

        unitree_ethernet = self.config.unitree_ethernet
        self.odom = UnitreeGo2OdomProvider(channel=unitree_ethernet)
        logging.info(f"Mapper Odom Provider: {self.odom}")

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _xy_to_latlon(self, x: float, y: float):
        φ0, λ0 = map(math.radians, (self.lat0, self.lon0))  # type: ignore
        φ = φ0 + y / R_EARTH
        λ = λ0 + x / (R_EARTH * math.cos(φ0))
        return map(math.degrees, (φ, λ))

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

    async def _poll(self) -> Optional[str]:
        """
        Poll for updated odometry and compute global pose.

        Returns
        -------
        Optional[str]
            Always returns None.
        """
        await asyncio.sleep(0.5)
        await self._update_pose()
        return None

    async def raw_to_text(self, raw_input: Optional[str]):
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

        message = self.buf[-1]
        self.buf.clear()
        self.io_provider.add_input(
            self.__class__.__name__, message.message, message.timestamp
        )
        return f"""
{self.descriptor_for_LLM} INPUT
// START
{message.message}
// END
"""
