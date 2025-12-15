import asyncio
import logging
import math

from actions.base import ActionConfig, ActionConnector
from providers.io_provider import IOProvider

from ..interface import MoveToPeerAction, MoveToPeerInput


class MoveToPeerRos2Connector(ActionConnector[ActionConfig, MoveToPeerInput]):
    """ROS 2 connector that turns toward the closest peer first, then drives.

    The algorithm:
    1. Retrieve current latitude/longitude (self) and closest‑peer lat/lon from
       IOProvider dynamic variables (populated by `GPSMagSerialReader`).
    2. Retrieve current yaw (° clockwise from geographic North) if available.
    3. Compute bearing from current position to peer (° clockwise from North).
    4. Compute heading error = (bearing − yaw) ∈ (‑180°, +180°].
    5. If the error is above a configurable tolerance (default ≈ 5 deg),
       rotate in place until within tolerance.
    6. Drive straight toward the peer until the separation < `STOP_DIST` m.

    All velocities are commanded in the body frame using Unitree's SportClient.
    """

    MAX_ROT_SPEED = 0.2  # rad/s (≈ 11.4 deg/s)
    FWD_SPEED = 0.4  # m/s forward once aligned
    ANG_TOL_DEG = 5.0  # degrees, acceptable pointing error
    STOP_DIST = 4.0  # metres to stop in front of peer

    def __init__(self, config: ActionConfig):
        """
        Initialize the MoveToPeerRos2Connector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)
        self.io = IOProvider()
        # defer heavy import; SportClient requires DDS initialisation
        from unitree.unitree_sdk2py.go2.sport.sport_client import SportClient

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

    async def connect(self, output_interface: MoveToPeerInput) -> None:
        """
        Execute the *navigate‑to‑peer* behaviour once.

        Parameters
        ----------
        output_interface : MoveToPeerInput
            The input protocol containing the action details.
        """
        if output_interface.action == MoveToPeerAction.IDLE:
            logging.info("MoveToPeer: idle, no movement commanded.")
            return

        lat0 = self.io.get_dynamic_variable("latitude")
        lon0 = self.io.get_dynamic_variable("longitude")
        lat1 = self.io.get_dynamic_variable("closest_peer_lat")
        lon1 = self.io.get_dynamic_variable("closest_peer_lon")
        yaw_var = self.io.get_dynamic_variable("yaw_deg")

        if None in (lat0, lon0):
            logging.info("MoveToPeer: own location not available, not moving.")
            return
        if None in (lat1, lon1):
            logging.info("MoveToPeer: peer location not available, not moving.")
            return

        lat0, lon0, lat1, lon1 = map(float, (lat0, lon0, lat1, lon1))
        yaw_deg = float(yaw_var) if yaw_var is not None else None

        R = 6_371_000.0  # Earth radius (m)
        dlat = math.radians(lat1 - lat0)
        dlon = math.radians(lon1 - lon0)
        x_east = dlon * math.cos(math.radians((lat0 + lat1) / 2)) * R  # +E
        y_north = dlat * R  # +N
        distance = math.hypot(x_east, y_north)

        if distance < self.STOP_DIST:
            logging.info(
                f"MoveToPeer: already near peer (d={distance:.1f} m < {self.STOP_DIST} m)."
            )
            return

        bearing_deg = math.degrees(math.atan2(x_east, y_north)) % 360.0

        if yaw_deg is None:
            logging.info("MoveToPeer: yaw unknown → driving body‑frame vector instead.")
            # project desired displacement into body frame *assuming* current
            # body‑frame aligns with world‑frame (worst‑case, but still works).
            norm = distance or 1.0
            vx = (y_north / norm) * self.FWD_SPEED
            vy = (x_east / norm) * self.FWD_SPEED
            self.sport_client.Move(vx, vy, 0.0)
            return

        # heading error – positive → need CW rotation
        heading_err = ((bearing_deg - yaw_deg + 180.0) % 360.0) - 180.0  # -> (‑180,180]
        logging.info(
            f"MoveToPeer: bearing={bearing_deg:.1f}°, yaw={yaw_deg:.1f}°, error={heading_err:.1f}°"
        )

        if abs(heading_err) > self.ANG_TOL_DEG:
            yaw_rate = math.copysign(self.MAX_ROT_SPEED, -heading_err)
            logging.info(f"MoveToPeer: rotating in place at {yaw_rate:.2f} rad/s")
            self.sport_client.Move(0.0, 0.0, yaw_rate)
            # give the robot time to rotate a bit before this action returns
            await asyncio.sleep(0.5)
            return  # caller may schedule subsequent actions for refinement

        logging.info(f"MoveToPeer: aligned → driving forward {self.FWD_SPEED} m/s")
        self.sport_client.Move(self.FWD_SPEED, 0.0, 0.0)
