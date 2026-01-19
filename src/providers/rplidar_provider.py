import json
import logging
import math
import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full
from typing import Dict, List, Optional, Union

import numpy as np
import zenoh
from numpy.typing import NDArray

from providers.odom_provider import OdomProvider
from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import LaserScan, open_zenoh_session, sensor_msgs

from .d435_provider import D435Provider
from .rplidar_driver import RPDriver
from .singleton import singleton


@dataclass
class RPLidarConfig:
    """
    Configuration for RPLidar.

    Parameters
    ----------
    max_buf_meas: int
        Maximum number of buffered measurements.
    min_len: int
        Minimum length of the scan.
    max_distance_mm: int
        Maximum distance in millimeters for valid measurements.
    """

    max_buf_meas: int = 0
    min_len: int = 5
    max_distance_mm: int = 10000


def rplidar_processor(
    data_queue: mp.Queue,
    control_queue: mp.Queue,
    serial_port: str,
    rplidar_config: RPLidarConfig,
    logging_config: Optional[LoggingConfig] = None,
):
    """
    Dedicated RPLidar processor function for multiprocessing.
    This function runs in a separate process to handle RPLidar data processing.

    Parameters
    ----------
    data_queue : mp.Queue
        Queue for receiving RPLidar data.
    control_queue : mp.Queue
        Queue for sending control commands.
    serial_port : str
        The name of the serial port in use by the RPLidar sensor.
    config : Dict
        Configuration dictionary containing parameters for the RPLidar.
    logging_config : Optional[LoggingConfig]
        Optional logging configuration. If provided, it will override the default logging settings.
    """
    setup_logging("rplidar_processor", logging_config=logging_config)

    running = True

    while running:
        lidar = None
        try:
            lidar = RPDriver(serial_port)

            info = lidar.get_info()
            logging.info(f"RPLidar Info: {info}")

            health = lidar.get_health()
            logging.info(f"RPLidar Health: {health[0]}")

            if health[0] != "Good":
                logging.error(f"There is a problem with the LIDAR: {health[0]}")
                time.sleep(0.5)
                continue

            lidar.reset()
            time.sleep(0.5)

            scan = lidar.iter_scans_local(
                scan_type="express",
                max_buf_meas=rplidar_config.max_buf_meas,
                min_len=rplidar_config.min_len,
                max_distance_mm=rplidar_config.max_distance_mm,
            )

            for scan_data in scan:
                try:
                    cmd = control_queue.get_nowait()
                    if cmd == "STOP":
                        running = False
                        break
                except Empty:
                    pass

                try:
                    data_queue.put_nowait(scan_data)
                except Full:
                    try:
                        data_queue.get_nowait()
                        data_queue.put_nowait(scan_data)
                    except (Empty, Full):
                        pass

        except Exception as e:
            logging.error(f"Error in RPLidar processor: {e}")
            if lidar:
                try:
                    lidar.reset()
                except Exception as reset_err:
                    logging.warning(f"Failed to reset RPLidar after error: {reset_err}")
            time.sleep(0.5)


@singleton
class RPLidarProvider:
    """
    RPLidar Provider.

    This class implements a singleton pattern to manage RPLidar data streaming.
    """

    # Constants
    DEFAULT_SERIAL_PORT = "/dev/cu.usbserial-0001"
    DEFAULT_HALF_WIDTH_ROBOT = 0.20
    DEFAULT_RELEVANT_DISTANCE_MAX = 1.1
    DEFAULT_RELEVANT_DISTANCE_MIN = 0.08
    DEFAULT_SENSOR_MOUNTING_ANGLE = 180.0
    NUM_BEZIER_POINTS = 10
    DEGREES_TO_RADIANS = math.pi / 180.0
    RADIANS_TO_DEGREES = 180.0 / math.pi

    def __init__(
        self,
        serial_port: str = DEFAULT_SERIAL_PORT,
        half_width_robot: float = DEFAULT_HALF_WIDTH_ROBOT,
        angles_blanked: Optional[list] = None,
        relevant_distance_max: float = DEFAULT_RELEVANT_DISTANCE_MAX,
        relevant_distance_min: float = DEFAULT_RELEVANT_DISTANCE_MIN,
        sensor_mounting_angle: float = DEFAULT_SENSOR_MOUNTING_ANGLE,
        URID: str = "",
        machine_type: str = "go2",
        use_zenoh: bool = False,
        simple_paths: bool = False,
        rplidar_config: RPLidarConfig = RPLidarConfig(),
        log_file: bool = False,
    ):
        """
        Initialize the RPLidar Provider with robot and sensor configuration.

        Parameters
        ----------
        serial_port: str = "/dev/cu.usbserial-0001"
            The name of the serial port in use by the RPLidar sensor.
        half_width_robot: float = 0.20
            The half width of the robot in m
        angles_blanked: list = []
            Regions of the scan to disregard, runs from -180 to +180 deg
        relevant_distance_max: float = 1.1
            Only consider barriers within this range, in m
        relevant_distance_min: float = 0.08
            Only consider barriers above this range, in m
        sensor_mounting_angle: float = 180.0
            The angle of the sensor zero relative to the way in which it's mounted
        URID: str = ""
            The URID of the robot, used for Zenoh communication
        machine_type: str = "go2"
            The type of the robot, e.g., "go2" or "tb4"
        use_zenoh: bool = False
            Whether to use Zenoh for communication
        simple_paths: bool = False
            Whether to use simple paths for path planning
        rplidar_config: RPLidarConfig = RPLidarConfig()
            Configuration for the RPLidar sensor
        log_file: bool = False
            Whether to log data to a local file
        """
        logging.info("Booting RPLidar")

        self.serial_port = serial_port
        self.half_width_robot = half_width_robot
        self.angles_blanked = angles_blanked if angles_blanked is not None else []
        self.relevant_distance_max = relevant_distance_max
        self.relevant_distance_min = relevant_distance_min
        self.sensor_mounting_angle = sensor_mounting_angle
        self.URID = URID
        self.machine_type = machine_type
        self.use_zenoh = use_zenoh
        self.simple_paths = simple_paths
        self.rplidar_config = rplidar_config
        self.log_file = log_file

        self.running: bool = False
        self.lidar = None
        self.zen = None
        self.scans = None

        self._raw_scan: Optional[NDArray] = None
        self._valid_paths: Optional[list] = None
        self._lidar_string: Optional[str] = None

        self.angles = None
        self.angles_final = None

        self.odom_rockchip_ts = 0.0
        self.odom_subscriber_ts = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw_m180_p180 = 0.0
        self.odom_yaw_0_360 = 0.0
        self.odom = OdomProvider()
        logging.info(f"Mapper Odom Provider: {self.odom}")

        self.write_to_local_file = False
        if log_file:
            self.write_to_local_file = log_file

        self.filename_current = None
        self.max_file_size_bytes = 1024 * 1024

        # Create timestamped log filename
        if self.write_to_local_file:
            self.filename_current = self.update_filename()
            logging.info(f"RPSCAN Logging to {self.filename_current}")

        # Initialize paths for path planning
        # Define 9 straight line paths separated by 15 degrees
        # Center path is 0° (straight forward), then ±15°, ±30°, ±45°, ±60°, 180° (backwards)
        self.path_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60, 180]
        self.paths = self._initialize_paths()

        self.pp = []
        for path in self.paths:
            pairs = list(zip(path[0], path[1]))
            self.pp.append(pairs)

        self.turn_left: List[int] = []
        self.turn_right: List[int] = []
        self.advance: List[int] = []
        self.retreat: bool = False

        self.data_queue = mp.Queue(maxsize=5)
        self.control_queue = mp.Queue()
        self._rplidar_processor_thread: Optional[mp.Process] = None

        self._serial_processor_thread: Optional[threading.Thread] = None

        if self.use_zenoh:
            logging.info("Connecting to the RPLIDAR via Zenoh")
            try:
                self.zen = open_zenoh_session()
                logging.info(f"Zenoh move client opened {self.zen}")

                if self.machine_type == "tb4":
                    logging.info(
                        f"{self.machine_type} RPLIDAR listener starting with URID: {self.URID}"
                    )
                    self.zen.declare_subscriber(
                        f"{self.URID}/pi/scan", self.listen_scan
                    )

                if self.machine_type == "go2":
                    logging.info(f"{self.machine_type} RPLIDAR listener starting")
                    self.zen.declare_subscriber("scan", self.listen_scan)

                if self.machine_type != "tb4" and self.machine_type != "go2":
                    raise ValueError(
                        f"Unsupported machine type: {self.machine_type}. Supported types are 'tb4' and 'go2'."
                    )

            except Exception as e:
                logging.error(f"Error opening Zenoh client: {e}")

        # D435 Provider
        self.d435_provider = D435Provider()

    def update_filename(self):
        """
        Update the filename for logging RPLidar data.
        """
        unix_ts = time.time()
        logging.info(f"RPSCAN time: {unix_ts}")
        unix_ts = str(unix_ts).replace(".", "_")
        filename = f"dump/lidar_{unix_ts}Z.jsonl"
        return filename

    def write_str_to_file(self, json_line: str):
        """
        Writes a dictionary to a file in JSON lines format. If the file exceeds max_file_size_bytes,
        creates a new file with a timestamp.

        Parameters
        ----------
        json_line : str
            The JSON string to write to the file.
        """
        if not isinstance(json_line, str):
            raise ValueError("Provided json_line must be a json string.")

        if (
            self.filename_current is not None
            and os.path.exists(self.filename_current)
            and os.path.getsize(self.filename_current) > self.max_file_size_bytes
        ):
            self.filename_current = self.update_filename()
            logging.info(f"New rpscan file name: {self.filename_current}")

        if self.filename_current is not None:
            with open(self.filename_current, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
                f.flush()

    def listen_scan(self, data: zenoh.Sample):
        """
        Zenoh scan handler.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing the scan data.
        """
        self.scans = sensor_msgs.LaserScan.deserialize(data.payload.to_bytes())
        logging.debug(f"Zenoh Laserscan data: {self.scans}")

        self._zenoh_processor(self.scans)

    def start(self):
        """
        Start the RPLidar provider.
        This method initializes the RPLidar processing thread and the serial data processing thread.
        """
        self.running = True

        if self.use_zenoh:
            logging.info("RPLidar using Zenoh, no serial port required")
            return

        if (
            not self._rplidar_processor_thread
            or not self._rplidar_processor_thread.is_alive()
        ):
            self._rplidar_processor_thread = mp.Process(
                target=rplidar_processor,
                args=(
                    self.data_queue,
                    self.control_queue,
                    self.serial_port,
                    self.rplidar_config,
                    get_logging_config(),
                ),
                daemon=True,
            )
            self._rplidar_processor_thread.start()

        if (
            not self._serial_processor_thread
            or not self._serial_processor_thread.is_alive()
        ):
            self._serial_processor_thread = threading.Thread(
                target=self._serial_processor, daemon=True
            )
            self._serial_processor_thread.start()
            logging.info("RPLidar processing thread started")

    def _zenoh_processor(self, scan: Optional[LaserScan]):
        """
        Preprocess Zenoh LaserScan data.

        Parameters
        ----------
        scan : Optional[LaserScan]
            The Zenoh LaserScan data to preprocess.
            If None, it indicates no data is available.
        """
        if scan is None:
            logging.info("Waiting for Zenoh Laserscan data...")
            self._raw_scan = None
            self._lidar_string = "You might be surrounded by objects and cannot safely move in any direction. DO NOT MOVE."
            self._valid_paths = []
        else:
            # logging.debug(f"_preprocess_zenoh: {scan}")
            # angle_min=-3.1241390705108643, angle_max=3.1415927410125732

            if not self.angles:
                self.angles = list(
                    map(
                        lambda x: 360.0 * (x + math.pi) / (2 * math.pi),
                        np.arange(scan.angle_min, scan.angle_max, scan.angle_increment),
                    )
                )
                self.angles_final = np.flip(np.array(self.angles))

            # angles now run from 360.0 to 0 degrees
            if self.angles_final is not None:
                data = list(zip(self.angles_final, scan.ranges))
            else:
                data = []
            array_ready = np.array(data)
            self._path_processor(array_ready)

    def _path_processor(self, data: NDArray):
        """
        Process the RPLidar data.
        This method processes the raw data from the RPLidar,
        filtering out irrelevant distances and angles,
        and converting the data into a format suitable for path planning.

        Parameters
        ----------
        data : NDArray
            The raw data from the RPLidar, expected to be a 2D array
            with angles and distances.
        """
        complexes = []
        raw = []

        for angle, distance in data:

            d_m = distance

            # first, correctly orient the sensor zero to the robot zero
            angle = angle + self.sensor_mounting_angle
            if angle >= 360.0:
                angle = angle - 360.0
            elif angle < 0.0:
                angle = 360.0 + angle

            raw.append([round(angle, 2), d_m])

            # don't worry about distant objects
            if d_m > self.relevant_distance_max:
                continue

            # don't worry about too close objects
            if d_m < self.relevant_distance_min:
                continue

            # convert the angle from [0 to 360] to [-180 to +180] range
            angle = angle - 180.0

            for b in self.angles_blanked:
                if angle >= b[0] and angle <= b[1]:
                    # this is a permanent robot reflection
                    # disregard
                    continue

            # Convert angle to radians for trigonometric calculations
            # Note: angle is adjusted back to [0, 360] range
            a_rad = (angle + 180.0) * self.DEGREES_TO_RADIANS

            v1 = d_m * math.cos(a_rad)
            v2 = d_m * math.sin(a_rad)

            # convert to x and y
            # x runs backwards to forwards, y runs left to right
            x = -1 * v2
            y = -1 * v1

            # the final data ready to use for path planning
            complexes.append([x, y, angle, d_m])

        # Append the D435 provider's obstacle data if available
        if self.d435_provider.running and len(self.d435_provider.obstacle) > 50:
            logging.debug("Appending D435 provider obstacle data to RPLidar data")
            for obstacle in self.d435_provider.obstacle:
                complexes.append(
                    [
                        obstacle["x"],
                        obstacle["y"],
                        obstacle["angle"],
                        obstacle["distance"],
                    ]
                )

        array = np.array(complexes)
        raw_array = np.array(raw)

        # save_timestamp = time.time()
        if self.write_to_local_file:
            try:
                json_line = json.dumps(
                    {
                        "odom_rockchip_ts": self.odom_rockchip_ts,
                        "odom_subscriber_ts": self.odom_subscriber_ts,
                        "odom_x": self.odom_x,
                        "odom_y": self.odom_y,
                        "odom_yaw_m180_p180": self.odom_yaw_m180_p180,
                        "odom_yaw_0_360": self.odom_yaw_0_360,
                        "frame": raw_array.tolist(),
                    }
                )
                self.write_str_to_file(json_line)
                logging.debug(f"rplidar wrote to: {self.filename_current}")
            except Exception as e:
                logging.error(f"Error saving rplidar to file: {str(e)}")

        # sort data into strictly increasing angles to deal with sensor issues
        # the sensor sometimes reports part of the previous scan and part of the next scan
        # so you end up with multiple slightly different values for some angles at the
        # junction

        """
        Determine set of possible paths
        """
        possible_paths = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        if self.simple_paths:
            # for the turtlebot - it can always turn in place,
            # only question is whether it can advance
            possible_paths = np.array([4])

        if array.ndim > 1:
            # we have valid LIDAR returns

            sorted_indices = array[:, 2].argsort()
            array = array[sorted_indices]

            # logging.debug(f"_process array: {array}")

            X = array[:, 0]
            Y = array[:, 1]
            # A = array[:, 2]
            D = array[:, 3]

            # all the possible conflicting points
            for x, y, d in list(zip(X, Y, D)):
                for apath in possible_paths:
                    path_points = self.paths[apath]
                    start_x, start_y = (
                        path_points[0][0],
                        path_points[1][0],
                    )
                    end_x, end_y = path_points[0][-1], path_points[1][-1]

                    if apath == 9:
                        # For going back, only consider obstacles that are:
                        # 1. Behind the robot (negative y in robot frame)
                        # 2. Within a certain distance threshold

                        # Assuming robot faces positive y direction
                        # Only check obstacles behind the robot
                        if y >= 0:  # Skip obstacles in front of or beside the robot
                            continue

                    dist_to_line = self.distance_point_to_line_segment(
                        x, y, start_x, start_y, end_x, end_y
                    )

                    if dist_to_line < self.half_width_robot:
                        # too close - this path will not work
                        # logging.info(f"removing path: {apath}")
                        path_to_remove = np.array([apath])
                        possible_paths = np.setdiff1d(possible_paths, path_to_remove)
                        logging.debug(f"remaining paths: {possible_paths}")
                        break  # no need to check other paths

        logging.info(f"possible_paths RP Lidar: {possible_paths}")

        self.turn_left = []
        self.turn_right = []
        self.advance = []
        self.retreat = False

        # convert to simple list
        ppl = possible_paths.tolist()

        for p in ppl:
            if p < 3:
                self.turn_left.append(p)
            elif p >= 3 and p <= 5:
                self.advance.append(p)
            elif p < 9:
                self.turn_right.append(p)
            elif p == 9:
                self.retreat = True

        return_string = self._generate_movement_string(ppl)

        self._raw_scan = array
        self._lidar_string = return_string
        self._valid_paths = ppl

        logging.debug(
            f"RPLidar Provider string: {self._lidar_string}\nValid paths: {self._valid_paths}"
        )

    def _serial_processor(self):
        """
        Serial data processing worker.

        This method works for the serial RPLidar driver without Zenoh.
        """
        while self.running:
            try:
                scan = self.data_queue.get_nowait()
                scan_array = np.array(scan)
                logging.debug(f"_serial_processor: {scan_array.ndim}")

                # the driver sends angles in degrees between from 0 to 360
                # warning - the driver may send two or more readings per angle,
                # this can be confusing for the code
                angles = scan_array[:, 0]

                logging.debug(f"_serial_processor: {angles}")

                # distances are in millimeters
                distances_m = scan_array[:, 1] / 1000

                data = list(zip(angles, distances_m))

                logging.debug(f"_serial_processor: {data}")
                array_ready = np.array(data)
                self._path_processor(array_ready)

                try:
                    o = self.odom.position
                    logging.debug(f"Odom data: {o}")
                    if o:
                        self.odom_x = o["odom_x"]
                        self.odom_y = o["odom_y"]
                        self.odom_rockchip_ts = o["odom_rockchip_ts"]
                        self.odom_subscriber_ts = o["odom_subscriber_ts"]
                        self.odom_yaw_m180_p180 = o["odom_yaw_m180_p180"]
                        self.odom_yaw_0_360 = o["odom_yaw_0_360"]
                except Exception as e:
                    logging.error(f"Error parsing Odom: {e}")

            except Empty:
                time.sleep(0.1)
                continue

    def stop(self):
        """
        Stop the RPLidar provider.
        """
        self.running = False

        if self._rplidar_processor_thread:
            logging.info("Stopping RPLidar processor thread")
            if not self.use_zenoh:
                self.control_queue.put("STOP")
                time.sleep(0.5)
            self._rplidar_processor_thread.join(timeout=5)

        if self._serial_processor_thread:
            logging.info("Stopping RPLidar serial processor thread")
            self._serial_processor_thread.join(timeout=5)

    @property
    def valid_paths(self) -> Optional[list]:
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
    def raw_scan(self) -> Optional[NDArray]:
        """
        Get the latest raw scan data.

        Returns
        -------
        Optional[NDArray]
            The latest raw scan result as a NumPy array, or None if not
            available.
        """
        return self._raw_scan

    @property
    def lidar_string(self) -> Optional[str]:
        """
        Get the latest natural language assessment of possible paths.

        Returns
        -------
        Optional[str]
            A natural language summary of possible motion paths
        """
        return self._lidar_string

    @property
    def movement_options(self) -> Dict[str, Union[List[int], bool]]:
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

    def _create_straight_path_from_angle(
        self, angle_degrees: float, length: float = 1.0, num_points: int = 30
    ) -> np.ndarray:
        """
        Create a straight path from a given angle.

        Parameters
        ----------
        angle_degrees : float
            The angle in degrees from which to create the path.
        length : float, optional
            The length of the path, by default 1.0.
        num_points : int, optional
            The number of points in the path, by default 30.

        Returns
        -------
        np.ndarray
            A NumPy array containing the x and y coordinates of the path points.
        """
        angle_rad = math.radians(angle_degrees)
        end_x = length * math.sin(angle_rad)
        end_y = length * math.cos(angle_rad)

        x_vals = np.linspace(0.0, end_x, num_points)
        y_vals = np.linspace(0.0, end_y, num_points)
        return np.array([x_vals, y_vals])

    def _initialize_paths(self) -> List[np.ndarray]:
        """
        Initialize paths for path planning.

        Returns
        -------
        List[np.ndarray]
            A list of NumPy arrays representing the paths.
        """
        return [
            self._create_straight_path_from_angle(angle, length=1.0)
            for angle in self.path_angles
        ]

    def distance_point_to_line_segment(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Calculate the distance from a point to a line segment.
        This method computes the shortest distance from a point (px, py) to a line segment defined by two endpoints (x1, y1) and (x2, y2).
        If the line segment has zero length, it returns the distance from the point to one of the endpoints.

        Parameters
        ----------
        px : float
            The x-coordinate of the point.
        py : float
            The y-coordinate of the point.
        x1 : float
            The x-coordinate of the first endpoint of the line segment.
        y1 : float
            The y-coordinate of the first endpoint of the line segment.
        x2 : float
            The x-coordinate of the second endpoint of the line segment.
        y2 : float
            The y-coordinate of the second endpoint of the line segment.

        Returns
        -------
        float
            The shortest distance from the point to the line segment.
            If the line segment has zero length, it returns the distance to the closest endpoint.
        """
        dx = x2 - x1
        dy = y2 - y1

        # If the line segment has zero length, return distance to point
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        # Calculate the parameter t that represents the projection of the point onto the line
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

        # Clamp t to [0, 1] to stay within the line segment
        t = max(0, min(1, t))

        # Find the closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

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

        if self.use_zenoh and self.machine_type == "tb4":  # TurtleBot4 control
            parts.append("'turn left', 'turn right', ")
            if self.advance:
                parts.append("'move forwards', ")
        else:
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
