import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_g1_navigation_provider import UnitreeG1NavigationProvider


class UnitreeG1Navigation(Background[BackgroundConfig]):
    """
    Background task for managing Unitree G1 robot navigation operations.

    This background task initializes and manages a UnitreeG1NavigationProvider
    that handles navigation goal publishing to the ROS2 Nav2 stack and monitors
    navigation status from the ROS2 action server. The provider automatically
    manages AI mode control during navigation operations.

    The navigation system is used for autonomous path planning, goal execution,
    and status monitoring in Unitree G1 robot applications. The provider runs
    as a singleton instance and maintains navigation state throughout the
    robot's operation lifecycle.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize the Unitree G1 Navigation background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background task. The navigation provider
            uses default settings and does not require additional configuration
            parameters beyond the base BackgroundConfig.

        Notes
        -----
        The provider is automatically started during initialization and will
        begin monitoring navigation status and managing navigation goals in
        the background. The provider uses a singleton pattern to ensure only
        one instance exists throughout the application lifecycle.
        """
        super().__init__(config)
        self.unitree_g1_navigation_provider = UnitreeG1NavigationProvider()
        self.unitree_g1_navigation_provider.start()
        logging.info("Unitree G1 Navigation Provider initialized in background")
