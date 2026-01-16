import logging

from backgrounds.base import Background, BackgroundConfig
from providers.unitree_go2_navigation_provider import UnitreeGo2NavigationProvider


class UnitreeGo2Navigation(Background[BackgroundConfig]):
    """
    Background task for monitoring Unitree Go2 robot navigation system.

    This class initializes and manages the UnitreeGo2NavigationProvider, which
    continuously monitors the navigation state of the Unitree Go2 robot. The
    navigation system provides real-time information about path planning, goal
    execution status, and navigation state transitions.

    The provider automatically starts in the background and maintains a connection
    to the robot's navigation system, enabling the agent to make informed decisions
    based on current navigation status during autonomous operations.

    Notes
    -----
    The navigation provider uses default settings and automatically starts when
    initialized. It operates as a singleton, ensuring only one instance manages
    the navigation state at a time.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize the Unitree Go2 Navigation background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background task. The navigation provider
            uses default settings and does not require additional configuration
            parameters.

        Notes
        -----
        The UnitreeGo2NavigationProvider is automatically started upon initialization
        and runs in the background. It maintains a connection to the robot's navigation
        system and provides real-time navigation state updates.
        """
        super().__init__(config)

        self.unitree_go2_navigation_provider = UnitreeGo2NavigationProvider()
        self.unitree_go2_navigation_provider.start()
        logging.info("Unitree Go2 Navigation Provider initialized in background")
