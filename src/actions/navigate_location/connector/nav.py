import asyncio
import logging

from actions.base import ActionConfig, ActionConnector
from actions.navigate_location.interface import NavigateLocationInput
from providers.io_provider import IOProvider
from providers.unitree_go2_locations_provider import UnitreeGo2LocationsProvider
from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider
from zenoh_msgs import Header, Point, Pose, PoseStamped, Quaternion, Time


class NavConnector(ActionConnector[NavigateLocationInput]):
    """
    Connector that queries a locations API and publishes a navigation goal.
    """

    def __init__(self, config: ActionConfig):
        """
        Initialize the NavConnector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        base_url = getattr(
            self.config,
            "base_url",
            "http://localhost:5000/maps/locations/list",
        )
        timeout = getattr(self.config, "timeout", 5)
        refresh_interval = getattr(self.config, "refresh_interval", 30)

        self.location_provider = UnitreeGo2LocationsProvider(
            base_url, timeout, refresh_interval
        )
        self.unitree_go2_navigation_provider = UnitreeGo2NavigationProvider()
        self.io_provider = IOProvider()

    async def connect(self, input_protocol: NavigateLocationInput) -> None:
        """
        Connect the input protocol to the navigation action.

        Parameters
        ----------
        input_protocol : NavigateLocationInput
            The input protocol containing the action details.
        """
        label = input_protocol.action

        label = label.lower().strip()
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

        # Use provider to lookup
        loc = self.location_provider.get_location(label)
        if loc is None:
            locations = self.location_provider.get_all_locations()
            locations_list = ", ".join(
                [
                    str(v.get("name") if isinstance(v, dict) else k)
                    for k, v in locations.items()
                ]
            )
            logging.warning(
                f"Location '{label}' not found. Available: {locations_list}"
                if locations_list
                else f"Location '{label}' not found. No locations available."
            )
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
            self.unitree_go2_navigation_provider.publish_goal_pose(goal_pose, label)
            logging.info(f"Navigation to '{label}' initiated")
        except Exception as e:
            logging.error(f"Error querying location list or publishing goal: {e}")
