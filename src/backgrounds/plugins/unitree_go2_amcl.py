import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_amcl_provider import UnitreeGo2AMCLProvider


class UnitreeGo2AMCL(Background[BackgroundConfig]):
    """
    Background task for reading AMCL (Adaptive Monte Carlo Localization) data from Unitree Go2 robot.

    This background task initializes and manages the UnitreeGo2AMCLProvider, which subscribes to
    AMCL pose data via Zenoh messaging. AMCL is a probabilistic localization algorithm that uses
    particle filters to estimate the robot's pose in a known map.

    The provider continuously monitors the robot's localization status and pose information,
    which is essential for autonomous navigation, path planning, and obstacle avoidance on the
    Unitree Go2 platform.

    Notes
    -----
    The provider automatically starts when this background task is initialized and runs
    as a singleton instance, ensuring only one AMCL provider exists per application.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize the Unitree Go2 AMCL background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background task. The AMCL provider uses default
            settings (topic: "amcl_pose", pose_tolerance: 0.4, yaw_tolerance: 0.2) and
            does not require additional configuration parameters.

        Notes
        -----
        The UnitreeGo2AMCLProvider is automatically started during initialization and will
        begin subscribing to AMCL pose messages via Zenoh. The provider runs as a singleton,
        so multiple instances of this background task will share the same provider instance.
        """
        super().__init__(config)

        self.unitree_go2_amcl_provider = UnitreeGo2AMCLProvider()
        self.unitree_go2_amcl_provider.start()
        logging.info("Unitree Go2 AMCL Provider initialized in background")
