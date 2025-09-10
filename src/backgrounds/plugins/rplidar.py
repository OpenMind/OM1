import logging

from backgrounds.base import Background, BackgroundConfig
from providers.rplidar_provider import RPLidarProvider


class RPLidar(Background):
    """
    Reads RPLidar data from RPLidar provider.
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        lidar_config = self._extract_lidar_config(config)

        self.lidar_provider = RPLidarProvider(**lidar_config)
        self.lidar_provider.start()
        logging.info("Initiated RPLidar Provider in background")

    def _extract_lidar_config(self, config: BackgroundConfig) -> dict:
        """
        Extract lidar configuration parameters from sensor config.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object containing lidar parameters.

        Returns
        -------
        dict
            Dictionary containing the extracted lidar configuration parameters.
        """
        lidar_config = {
            "serial_port": getattr(config, "serial_port", None),
            "use_zenoh": getattr(config, "use_zenoh", False),
            "half_width_robot": getattr(config, "half_width_robot", 0.20),
            "angles_blanked": getattr(config, "angles_blanked", []),
            "relevant_distance_max": getattr(config, "relevant_distance_max", 1.1),
            "relevant_distance_min": getattr(config, "relevant_distance_min", 0.08),
            "sensor_mounting_angle": getattr(config, "sensor_mounting_angle", 180.0),
            "URID": getattr(config, "URID", ""),
            "multicast_address": getattr(config, "multicast_address", ""),
            "machine_type": getattr(config, "machine_type", "go2"),
            "log_file": getattr(config, "log_file", False),
        }

        return lidar_config
