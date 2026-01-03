import logging
from typing import List, Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.rplidar_provider import RPLidarProvider


class RPLidarConfig(BackgroundConfig):
    """
    Configuration for RPLidar Background.

    Parameters
    ----------
    serial_port : Optional[str]
        Serial port for the RPLidar device.
    use_zenoh : bool
        Whether to use Zenoh.
    half_width_robot : float
        Half width of the robot in meters.
    angles_blanked : List[float]
        Angles to blank out from lidar scan.
    relevant_distance_max : float
        Maximum relevant distance in meters.
    relevant_distance_min : float
        Minimum relevant distance in meters.
    sensor_mounting_angle : float
        Sensor mounting angle in degrees.
    URID : str
        Unique Robot ID.
    machine_type : str
        Type of machine.
    log_file : bool
        Whether to log to file.
    """

    serial_port: Optional[str] = Field(
        default=None, description="Serial port for the RPLidar device"
    )
    use_zenoh: bool = Field(default=False, description="Whether to use Zenoh")
    half_width_robot: float = Field(
        default=0.20, description="Half width of the robot in meters"
    )
    angles_blanked: List[float] = Field(
        default_factory=list, description="Angles to blank out from lidar scan"
    )
    relevant_distance_max: float = Field(
        default=1.1, description="Maximum relevant distance in meters"
    )
    relevant_distance_min: float = Field(
        default=0.08, description="Minimum relevant distance in meters"
    )
    sensor_mounting_angle: float = Field(
        default=180.0, description="Sensor mounting angle in degrees"
    )
    URID: str = Field(default="", description="Unique Robot ID")
    machine_type: str = Field(default="go2", description="Type of machine")
    log_file: bool = Field(default=False, description="Whether to log to file")


class RPLidar(Background[RPLidarConfig]):
    """
    Reads RPLidar data from RPLidar provider.
    """

    def __init__(self, config: RPLidarConfig):
        super().__init__(config)

        lidar_config = {
            "serial_port": self.config.serial_port,
            "use_zenoh": self.config.use_zenoh,
            "half_width_robot": self.config.half_width_robot,
            "angles_blanked": self.config.angles_blanked,
            "relevant_distance_max": self.config.relevant_distance_max,
            "relevant_distance_min": self.config.relevant_distance_min,
            "sensor_mounting_angle": self.config.sensor_mounting_angle,
            "URID": self.config.URID,
            "machine_type": self.config.machine_type,
            "log_file": self.config.log_file,
        }

        self.lidar_provider = RPLidarProvider(**lidar_config)
        self.lidar_provider.start()
        logging.info("Initiated RPLidar Provider in background")
