import inspect
import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

from zenoh_idl.geometry_msgs import Pose, PoseStamped

from .function_call_provider import FunctionGenerator, LLMFunction
from .singleton import singleton
from .unitree_go2_amcl_provider import UnitreeGo2AMCLProvider
from .unitree_go2_navigation_provider import UnitreeGo2NavigationProvider


@singleton
class UnitreeGo2LocationProvider:
    """
    Location Provider for Unitree Go2 robot that can be used as function calls for LLM.
    Provides functionality to record, retrieve, and navigate to saved locations.
    Uses existing NavigationProvider and AMCLProvider for core functionality.
    """

    def __init__(
        self,
        locations_file: str = "saved_locations.json",
    ):
        """
        Initialize the Unitree Go2 Location Provider.

        Parameters
        ----------
        locations_file : str, optional
            The file to store saved locations (default is "saved_locations.json").
        """
        self.navigation_provider = UnitreeGo2NavigationProvider()
        self.amcl_provider = UnitreeGo2AMCLProvider()

        self.locations_file = locations_file

        self.saved_locations: Dict[str, Dict] = self._load_locations()

        self.running: bool = False

        logging.info("Location Provider initialized")

    def _load_locations(self) -> Dict[str, Dict]:
        """
        Load saved locations from file.

        Returns
        -------
        Dict[str, Dict]
            Dictionary containing saved locations.
        """
        if os.path.exists(self.locations_file):
            try:
                with open(self.locations_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading locations file: {e}")
                return {}
        return {}

    def _save_locations(self):
        """
        Save locations to file.
        """
        try:
            with open(self.locations_file, "w") as f:
                json.dump(self.saved_locations, f, indent=2)
            logging.info(f"Saved locations to {self.locations_file}")
        except Exception as e:
            logging.error(f"Error saving locations file: {e}")

    def start(self):
        """
        Start the location provider by starting the underlying providers.
        """
        if self.running:
            logging.warning("Location Provider is already running")
            return

        self.navigation_provider.start()
        self.amcl_provider.start()

        self.running = True

        logging.info("Location Provider started")

    def generate_llm_functions(self) -> Dict:
        """
        Generate OpenAI function schemas for all decorated methods.

        Returns
        -------
        Dict
            Dictionary containing function schemas for LLM.
        """
        return FunctionGenerator.generate_functions_from_class(self)

    def get_llm_function_mapping(self) -> Dict:
        """
        Get mapping of function names to actual methods for execution.

        Returns
        -------
        Dict
            Dictionary mapping function names to their corresponding methods.
        """
        mapping = {}
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_llm_function") and method._llm_function:
                mapping[method._llm_name] = method
        return mapping

    @LLMFunction("Get the robot's current location and localization status")
    def get_current_location(self) -> Dict:
        """
        Get the current location of the robot.

        Returns
        -------
        Dict
            Dictionary containing current pose information and localization status.
        """
        if not self.amcl_provider.is_localized:
            return {
                "success": False,
                "message": "Robot is not properly localized",
                "localization_status": self.amcl_provider.is_localized,
                "pose": None,
            }

        current_pose = self.amcl_provider.pose
        if current_pose is None:
            return {
                "success": False,
                "message": "No current pose available",
                "localization_status": self.amcl_provider.is_localized,
                "pose": None,
            }

        pose_dict = {
            "position": {
                "x": current_pose.position.x,
                "y": current_pose.position.y,
                "z": current_pose.position.z,
            },
            "orientation": {
                "x": current_pose.orientation.x,
                "y": current_pose.orientation.y,
                "z": current_pose.orientation.z,
                "w": current_pose.orientation.w,
            },
        }

        return {
            "success": True,
            "message": "Current location retrieved successfully",
            "localization_status": self.amcl_provider.is_localized,
            "pose": pose_dict,
        }

    @LLMFunction(
        "Save the robot's current location with a name and optional description"
    )
    def record_location(self, location_name: str, description: str = "") -> Dict:
        """
        Record the current location with a given name.

        Parameters
        ----------
        location_name : str
            Name to assign to the current location.
        description : str, optional
            Optional description of the location.

        Returns
        -------
        Dict
            Dictionary containing success status and message.
        """
        if not self.amcl_provider.is_localized:
            return {
                "success": False,
                "message": "Cannot record location: Robot is not properly localized",
            }

        current_pose = self.amcl_provider.pose
        if current_pose is None:
            return {
                "success": False,
                "message": "Cannot record location: No current pose available",
            }

        location_data = {
            "name": location_name,
            "description": description,
            "pose": {
                "position": {
                    "x": current_pose.position.x,
                    "y": current_pose.position.y,
                    "z": current_pose.position.z,
                },
                "orientation": {
                    "x": current_pose.orientation.x,
                    "y": current_pose.orientation.y,
                    "z": current_pose.orientation.z,
                    "w": current_pose.orientation.w,
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

        self.saved_locations[location_name] = location_data
        self._save_locations()

        return {
            "success": True,
            "message": f"Location '{location_name}' recorded successfully",
            "location_data": location_data,
        }

    @LLMFunction("Get all saved locations")
    def get_saved_locations(self) -> Dict:
        """
        Get all saved locations.

        Returns
        -------
        Dict
            Dictionary containing all saved locations.
        """
        return {
            "success": True,
            "message": f"Retrieved {len(self.saved_locations)} saved locations",
            "locations": self.saved_locations,
        }

    @LLMFunction("Get detailed information about a specific saved location")
    def get_location_info(self, location_name: str) -> Dict:
        """
        Get information about a specific saved location.

        Parameters
        ----------
        location_name : str
            Name of the location to retrieve.

        Returns
        -------
        Dict
            Dictionary containing location information.
        """
        if location_name not in self.saved_locations:
            return {
                "success": False,
                "message": f"Location '{location_name}' not found",
            }

        return {
            "success": True,
            "message": f"Location '{location_name}' information retrieved",
            "location_data": self.saved_locations[location_name],
        }

    @LLMFunction("Command the robot to navigate to a saved location")
    def navigate_to_location(self, location_name: str) -> Dict:
        """
        Navigate to a saved location.

        Parameters
        ----------
        location_name : str
            Name of the location to navigate to.

        Returns
        -------
        Dict
            Dictionary containing navigation command status.
        """
        if location_name not in self.saved_locations:
            return {
                "success": False,
                "message": f"Location '{location_name}' not found",
            }

        location_data = self.saved_locations[location_name]
        pose_data = location_data["pose"]

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp.sec = int(datetime.now().timestamp())
        goal_pose.header.stamp.nanosec = int((datetime.now().timestamp() % 1) * 1e9)

        goal_pose.pose.position.x = pose_data["position"]["x"]
        goal_pose.pose.position.y = pose_data["position"]["y"]
        goal_pose.pose.position.z = pose_data["position"]["z"]

        goal_pose.pose.orientation.x = pose_data["orientation"]["x"]
        goal_pose.pose.orientation.y = pose_data["orientation"]["y"]
        goal_pose.pose.orientation.z = pose_data["orientation"]["z"]
        goal_pose.pose.orientation.w = pose_data["orientation"]["w"]

        try:
            self.navigation_provider.publish_goal_pose(goal_pose)
            logging.info(f"Navigation to location '{location_name}' initiated")

            return {
                "success": True,
                "message": f"Navigation to location '{location_name}' initiated",
                "target_location": location_data,
            }
        except Exception as e:
            logging.error(f"Error initiating navigation: {e}")
            return {
                "success": False,
                "message": f"Error initiating navigation: {str(e)}",
            }

    @LLMFunction("Delete a saved location")
    def delete_location(self, location_name: str) -> Dict:
        """
        Delete a saved location.

        Parameters
        ----------
        location_name : str
            Name of the location to delete.

        Returns
        -------
        Dict
            Dictionary containing deletion status.
        """
        if location_name not in self.saved_locations:
            return {
                "success": False,
                "message": f"Location '{location_name}' not found",
            }

        deleted_location = self.saved_locations.pop(location_name)
        self._save_locations()

        return {
            "success": True,
            "message": f"Location '{location_name}' deleted successfully",
            "deleted_location": deleted_location,
        }

    @LLMFunction("Get current navigation and localization status")
    def get_navigation_status(self) -> Dict:
        """
        Get current navigation status.

        Returns
        -------
        Dict
            Dictionary containing navigation status.
        """
        return {
            "success": True,
            "message": "Navigation status retrieved",
            "navigation_status": self.navigation_provider.navigation_state,
            "localization_status": self.amcl_provider.is_localized,
        }

    @LLMFunction("Get a list of all saved location names")
    def list_location_names(self) -> Dict:
        """
        Get a list of all saved location names.

        Returns
        -------
        Dict
            Dictionary containing list of location names.
        """
        location_names = list(self.saved_locations.keys())
        return {
            "success": True,
            "message": f"Found {len(location_names)} saved locations",
            "location_names": location_names,
        }

    @LLMFunction("Calculate the distance from current position to a saved location")
    def get_distance_to_location(self, location_name: str) -> Dict:
        """
        Calculate approximate distance to a saved location.

        Parameters
        ----------
        location_name : str
            Name of the location to calculate distance to.

        Returns
        -------
        Dict
            Dictionary containing distance information.
        """
        if location_name not in self.saved_locations:
            return {
                "success": False,
                "message": f"Location '{location_name}' not found",
            }

        if not self.amcl_provider.is_localized:
            return {
                "success": False,
                "message": "Cannot calculate distance: Robot is not properly localized",
            }

        current_pose = self.amcl_provider.pose
        if current_pose is None:
            return {
                "success": False,
                "message": "Cannot calculate distance: No current pose available",
            }

        target_pose = self.saved_locations[location_name]["pose"]

        dx = current_pose.position.x - target_pose["position"]["x"]
        dy = current_pose.position.y - target_pose["position"]["y"]
        distance = (dx**2 + dy**2) ** 0.5

        return {
            "success": True,
            "message": f"Distance to '{location_name}' calculated",
            "distance_meters": distance,
            "current_position": {
                "x": current_pose.position.x,
                "y": current_pose.position.y,
            },
            "target_position": {
                "x": target_pose["position"]["x"],
                "y": target_pose["position"]["y"],
            },
        }

    @LLMFunction("Update the description of a saved location")
    def update_location_description(
        self, location_name: str, new_description: str
    ) -> Dict:
        """
        Update the description of a saved location.

        Parameters
        ----------
        location_name : str
            Name of the location to update.
        new_description : str
            New description for the location.

        Returns
        -------
        Dict
            Dictionary containing update status.
        """
        if location_name not in self.saved_locations:
            return {
                "success": False,
                "message": f"Location '{location_name}' not found",
            }

        old_description = self.saved_locations[location_name]["description"]
        self.saved_locations[location_name]["description"] = new_description
        self.saved_locations[location_name]["last_updated"] = datetime.now().isoformat()
        self._save_locations()

        return {
            "success": True,
            "message": f"Description for '{location_name}' updated successfully",
            "old_description": old_description,
            "new_description": new_description,
        }

    @property
    def is_localized(self) -> bool:
        """Check if the robot is localized."""
        return self.amcl_provider.is_localized

    @property
    def current_navigation_status(self) -> str:
        """Get the current navigation status."""
        return self.navigation_provider.navigation_state

    @property
    def current_pose(self) -> Optional[Pose]:
        """Get the current pose."""
        return self.amcl_provider.pose

    @property
    def location_count(self) -> int:
        """Get the number of saved locations."""
        return len(self.saved_locations)
