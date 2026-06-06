import asyncio
import logging

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.dock_charging.interface import DockChargingInput
from providers.unitree_go2_charging_provider import UnitreeGo2ChargingProvider
from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider
from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider
from zenoh_msgs import Header, Point, Pose, PoseStamped, Quaternion, Time

# Charging status codes from UnitreeGo2ChargingProvider
# 0 = not charging / not docked
# 1 = charging / docked
# Verify this value against your Unitree Go2 SDK if different.
CHARGING_STATUS_DOCKED = 1


class UnitreeGo2DockConfig(ActionConfig):
    """
    Configuration for Unitree Go2 Dock Charging connector.

    Parameters
    ----------
    base_url : str
        The base URL for the locations API.
    timeout : int
        Timeout for the HTTP requests in seconds.
    refresh_interval : int
        Interval to refresh the locations list in seconds.
    default_dock_location : str
        Default saved location name for the charging dock.
        Must match a location previously saved with remember_location.
    """

    base_url: str = Field(
        default="http://localhost:5000/maps/locations/list",
        description="The base URL for the locations API.",
    )
    timeout: int = Field(
        default=5,
        description="Timeout for the HTTP requests in seconds.",
    )
    refresh_interval: int = Field(
        default=30,
        description="Interval to refresh the locations list in seconds.",
    )
    default_dock_location: str = Field(
        default="charging_dock",
        description="Default saved location name for the charging dock.",
    )


class UnitreeGo2DockConnector(ActionConnector[UnitreeGo2DockConfig, DockChargingInput]):
    """
    Dock charging connector for Unitree Go2 robots.

    Navigates the robot to its saved charging dock location.
    Skips navigation if the robot is already docked and charging.

    Prerequisites
    -------------
    The charging dock location must have been previously saved using
    the remember_location action with a name matching 'dock_location_name'
    (default: "charging_dock").
    """

    def __init__(self, config: UnitreeGo2DockConfig):
        """
        Initialize the UnitreeGo2DockConnector.

        Parameters
        ----------
        config : UnitreeGo2DockConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.location_provider = UnitreeGo2LocationsProvider(
            self.config.base_url,
            self.config.timeout,
            self.config.refresh_interval,
        )
        self.navigation_provider = UnitreeGo2NavigationProvider()
        self.charging_provider = UnitreeGo2ChargingProvider()
        self.charging_provider.start()

        logging.info(
            "[DockGo2Connector] Initialized. Default dock location: '%s'",
            self.config.default_dock_location,
        )

    async def connect(self, output_interface: DockChargingInput) -> None:
        """
        Execute the dock charging action.

        Steps:
        1. Skip if robot is already docked and charging.
        2. Resolve dock location name from input or config default.
        3. Look up dock coordinates from saved locations.
        4. Build and publish navigation goal pose.

        Parameters
        ----------
        output_interface : DockChargingInput
            Input containing dock command and optional dock location name.
        """
        # Step 1: Skip if already docked
        current_status = self.charging_provider.get_charging_status()
        if current_status == CHARGING_STATUS_DOCKED:
            logging.info(
                "[DockGo2Connector] Robot is already docked and charging. No action needed."
            )
            return

        # Step 2: Resolve dock location name
        dock_label = (
            output_interface.dock_location_name or self.config.default_dock_location
        )
        dock_label = dock_label.lower().strip()

        logging.info("[DockGo2Connector] Navigating to charging dock: '%s'", dock_label)

        # Step 3: Look up dock coordinates from saved locations
        loc = self.location_provider.get_location(dock_label)
        if loc is None:
            locations = self.location_provider.get_all_locations()
            locations_list = ", ".join(
                str(v.get("name") if isinstance(v, dict) else k)
                for k, v in locations.items()
            )
            msg = (
                f"Dock location '{dock_label}' not found. Available: {locations_list}"
                if locations_list
                else f"Dock location '{dock_label}' not found. No locations available."
            )
            logging.warning("[DockGo2Connector] %s", msg)
            return

        # Step 4: Build and publish navigation goal pose
        pose = loc.get("pose") or {}
        position = pose.get("position", {})
        orientation = pose.get("orientation", {})

        now = Time(sec=int(asyncio.get_event_loop().time()), nanosec=0)
        header = Header(stamp=now, frame_id="map")
        position_msg = Point(
            x=float(position.get("x", 0.0)),
            y=float(position.get("y", 0.0)),
            z=float(position.get("z", 0.0)),
        )
        orientation_msg = Quaternion(
            x=float(orientation.get("x", 0.0)),
            y=float(orientation.get("y", 0.0)),
            z=float(orientation.get("z", 0.0)),
            w=float(orientation.get("w", 1.0)),
        )
        pose_msg = Pose(position=position_msg, orientation=orientation_msg)
        goal_pose = PoseStamped(header=header, pose=pose_msg)

        try:
            self.navigation_provider.publish_goal_pose(goal_pose, dock_label)
            logging.info(
                "[DockGo2Connector] Navigation to dock '%s' initiated successfully.",
                dock_label,
            )
        except Exception as e:
            logging.error(
                "[DockGo2Connector] Failed to publish navigation goal to dock '%s': %s",
                dock_label,
                e,
            )
