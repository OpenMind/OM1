import logging
import zenoh
from typing import Dict, List, Optional

from zenoh_idl import sensor_msgs

from .singleton import singleton

@singleton
class SimplePathsProvider:
    def __init__(self):
        self.session = None
        self.paths = None

        # Paths attributes
        self.turn_left = []
        self.turn_right = []
        self.advance = []
        self.retreat = False

        # Valid paths list
        self._valid_paths = []

        # LLM string
        self._lidar_string = ""

        # Path angles for movement options
        self.path_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]

        try:
            self.session = zenoh.open(zenoh.Config())
            self.session.declare_subscriber(
                "om/paths", self.paths_callback
            )
            logging.info("Zenoh is open for SimplePathProvider")
        except Exception as e:
            logging.error(f"Failed to open Zenoh session: {e}")

    def start(self):
        """
        Start the SimplePathsProvider by opening a Zenoh session.
        """
        pass

    def stop(self):
        """
        Stop the SimplePathsProvider by closing the Zenoh session.
        """
        if self.session:
            self.session.close()
            logging.info("Zenoh session closed.")
        else:
            logging.warning("No Zenoh session to close.")

    def paths_callback(self, msg: zenoh.Sample):
        """
        Callback for receiving paths messages.

        Parameters:
        -----------
        msg: zenoh.Sample
            The message containing paths data.
        """
        self.paths = sensor_msgs.Paths.deserialize(msg.payload.to_bytes())
        logging.info(f"Received paths: {self.paths.paths}")

        self.turn_left = []
        self.turn_right = []
        self.advance = []
        self.retreat = False

        for path in self.paths.paths:
            if path < 3:
                self.turn_left.append(path)
            elif path >= 3 and path <= 5:
                self.advance.append(path)
            elif path < 9:
                self.turn_right.append(path)
            elif path == 9:
                self.retreat = True

        self._valid_paths = self.paths.paths.copy()
        self._lidar_string = self._generate_movement_string(self.paths.paths)

    def _generate_movement_string(self, valid_paths: list) -> str:
        """
        Generate movement direction string based on valid paths.

        Parameters
        ----------
        valid_paths : list
            A list of valid paths represented as integers.
            Each integer corresponds to a specific movement direction.

        Returns
        -------
        str
            A string describing the safe movement directions based on the valid paths.
        """
        if not valid_paths:
            return "You are surrounded by objects and cannot safely move in any direction. DO NOT MOVE."

        parts = ["The safe movement directions are: {"]

        if self.turn_left:
            parts.append("'turn left', ")
        if self.advance:
            parts.append("'move forwards', ")
        if self.turn_right:
            parts.append("'turn right', ")
        if self.retreat:
            parts.append("'move back', ")

        parts.append("'stand still'}. ")
        return "".join(parts)


    @property
    def valid_paths(self) -> Optional[List]:
        """
        Get the currently valid paths.

        Returns
        -------
        Optional[list]
            The currently valid paths as a list, or None if not
            available. The list contains 0 to 10 entries,
            corresponding to possible paths - for example: [0,3,4,5]
        """
        return self._valid_paths

    @property
    def lidar_string(self) -> str:
        """
        Get the latest natural language assessment of possible paths.

        Returns
        -------
        str
            A natural language summary of possible motion paths
        """
        return self._lidar_string

    @property
    def movement_options(self) -> Dict[str, List[int]]:
        """
        Get the movement options based on the current valid paths.

        Returns
        -------
        Dict[str, List[int]]
            A dictionary containing lists of valid movement options:
            - 'turn_left': Indices for turning left
            - 'advance': Indices for moving forward
            - 'turn_right': Indices for turning right
            - 'retreat': Indices for moving backward
        """
        return {
            "turn_left": self.turn_left,
            "advance": self.advance,
            "turn_right": self.turn_right,
            "retreat": self.retreat,
        }
