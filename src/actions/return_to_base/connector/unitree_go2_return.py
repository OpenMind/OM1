import asyncio
import logging

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.return_to_base.interface import ReturnToBaseInput
from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider
from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider
from zenoh_msgs import Header, Point, Pose, PoseStamped, Quaternion, Time


class UnitreeGo2ReturnToBaseConfig(ActionConfig):
    """
    Configuration for Unitree Go2 Return To Base connector.

    Parameters
    ----------
    base_url : str
        The base URL for the locations API.
    timeout : int
        Timeout for the HTTP requests in seconds.
    refresh_interval : int
        Interval to refresh the locations list in seconds.
    default_base_location : str
        Default saved location name for the base.
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
    default_base_location: str = Field(
        default="base",
        description="Default saved location name for the base.",
    )


class UnitreeGo2ReturnToBaseConnector(
    ActionConnector[UnitreeGo2ReturnToBaseConfig, ReturnToBaseInput]
):
    """
    Return to base connector for Unitree Go2 robots.

    Navigates the robot back to its saved base location.

    Prerequisites
    -------------
    The base location must have been previously saved using
    the remember_location action with a name matching 'base_location_name'
    (default: "base").
    """

    def __init__(self, config: UnitreeGo2ReturnToBaseConfig):
        """
        Initialize the UnitreeGo2ReturnToBaseConnector.

        Parameters
        ----------
        config : UnitreeGo2ReturnToBaseConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.location_provider = UnitreeGo2LocationsProvider(
            self.config.base_url,
            self.config.timeout,
            self.config.refresh_interval,
        )
        self.navigation_provider = UnitreeGo2NavigationProvider()

        logging.info(
            "[ReturnToBaseGo2Connector] Initialized. Default base location: '%s'",
            self.config.default_base_location,
        )

    async def connect(self, output_interface: ReturnToBaseInput) -> None:
        """
        Execute the return to base action.

        Steps:
        1. Resolve base location name from input or config default.
        2. Look up base coordinates from saved locations.
        3. Build and publish navigation goal pose.

        Parameters
        ----------
        output_interface : ReturnToBaseInput
            Input containing return command and optional base location name.
        """
        # Step 1: Resolve base location name
        base_label = (
            output_interface.base_location_name or self.config.default_base_location
        )
        base_label = base_label.lower().strip()

        logging.info("[ReturnToBaseGo2Connector] Navigating to base: '%s'", base_label)

        # Step 2: Look up base coordinates from saved locations
        loc = self.location_provider.get_location(base_label)
        if loc is None:
            locations = self.location_provider.get_all_locations()
            locations_list = ", ".join(
                str(v.get("name") if isinstance(v, dict) else k)
                for k, v in locations.items()
            )
            msg = (
                f"Base location '{base_label}' not found. Available: {locations_list}"
                if locations_list
                else f"Base location '{base_label}' not found. No locations available."
            )
            logging.warning("[ReturnToBaseGo2Connector] %s", msg)
            return

        # Step 3: Build and publish navigation goal pose
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
            self.navigation_provider.publish_goal_pose(goal_pose, base_label)
            logging.info(
                "[ReturnToBaseGo2Connector] Navigation to base '%s' initiated successfully.",
                base_label,
            )
        except Exception as e:
            logging.error(
                "[ReturnToBaseGo2Connector] Failed to publish navigation goal to base '%s': %s",
                base_label,
                e,
            )
