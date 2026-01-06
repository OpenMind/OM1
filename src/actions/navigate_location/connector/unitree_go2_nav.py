import asyncio
import logging

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.navigate_location.interface import NavigateLocationInput
from providers.io_provider import IOProvider
from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider
from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider
from zenoh_msgs import Header, Point, Pose, PoseStamped, Quaternion, Time


class UnitreeGo2NavConfig(ActionConfig):
    """
    Configuration for Unitree Go2 Navigation connector.

    Parameters
    ----------
    base_url : str
        The base URL for the locations API.
    timeout : int
        Timeout for the HTTP requests in seconds.
    refresh_interval : int
        Interval to refresh the locations list in seconds.
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


class UnitreeGo2NavConnector(
    ActionConnector[UnitreeGo2NavConfig, NavigateLocationInput]
):
    """
    Navigation/location connector for Unitree Go2 robots.
    """

    def __init__(self, config: UnitreeGo2NavConfig):
        """
        Initialize the UnitreeGo2NavConnector.

        Parameters
        ----------
        config : UnitreeGo2NavConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        base_url = self.config.base_url
        timeout = self.config.timeout
        refresh_interval = self.config.refresh_interval

        self.location_provider = UnitreeGo2LocationsProvider(
            base_url, timeout, refresh_interval
        )
        self.navigation_provider = UnitreeGo2NavigationProvider()
        self.io_provider = IOProvider()
        logging.info(
            "[NavGo2Connector] Using UnitreeGo2 providers for locations and navigation."
        )

    async def connect(self, output_interface: NavigateLocationInput) -> None:
        """
        Connect the input protocol to the navigate location action for Go2.

        Parameters
        ----------
        output_interface : NavigateLocationInput
            The input protocol containing the action details.
        """
        label = output_interface.action.lower().strip()
        for prefix in [
            "go to the ",
            "go to ",
            "navigate to the ",
            "navigate to ",
            "move to the ",
            "move to ",
            "take me to the ",
            "take me to ",
        ]:
            if label.startswith(prefix):
                label = label[len(prefix) :].strip()
                logging.info(
                    f"Cleaned location label: removed '{prefix}' prefix -> '{label}'"
                )
                break

        loc = self.location_provider.get_location(label)
        if loc is None:
            locations = self.location_provider.get_all_locations()
            locations_list = ", ".join(
                str(v.get("name") if isinstance(v, dict) else k)
                for k, v in locations.items()
            )
            msg = (
                f"Location '{label}' not found. Available: {locations_list}"
                if locations_list
                else f"Location '{label}' not found. No locations available."
            )
            logging.warning(msg)
            return

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
            self.navigation_provider.publish_goal_pose(goal_pose, label)
            logging.info(f"Navigation to '{label}' initiated")
        except Exception as e:
            logging.error(f"Error querying location list or publishing goal: {e}")
