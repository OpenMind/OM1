import json
import logging
from typing import Callable, Dict, Optional

import zenoh

from zenoh_msgs import String

from .context_provider import ContextProvider
from .prometheus_monitor import PrometheusMonitor
from .singleton import singleton
from .zenoh_listener_provider import ZenohListenerProvider


@singleton
class UnitreeGo2FrontierExplorationProvider(ZenohListenerProvider):
    """
    Frontier Exploration provider for Unitree Go2 robot.
    """

    def __init__(
        self,
        topic: str = "explore/status",
        context_aware_text: Dict = {"exploration_done": True},
    ):
        """
        Initialize the Frontier Exploration Provider with a specific topic.

        Parameters
        ----------
        topic : str, optional
            The topic on which to subscribe for frontier exploration messages (default is "explore/status").
        context_aware_text : Dict, optional
            The context text to be sent when exploration is complete. Defaults to {"exploration_done": True}.
        """
        super().__init__(topic)
        logging.info("Frontier Exploration Provider initialized with topic: %s", topic)

        self.exploration_info: Optional[str] = None
        self.exploration_complete = False

        # Start Context Provider to update context when exploration is done
        self.context_provider = ContextProvider()
        self.context_aware_text = context_aware_text

        # Register with Prometheus monitor
        self._monitor = PrometheusMonitor()
        self._monitor.register(
            "UnitreeGo2FrontierExplorationProvider",
            metadata={"type": "frontier_exploration", "robot": "unitree_go2"},
            recovery_callback=self._recover,
        )

    def frontier_exploration_message_callback(self, data: zenoh.Sample):
        """
        Process an incoming frontier exploration message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        if data.payload:
            message: str = String.deserialize(data.payload.to_bytes()).data

            try:
                exploration_data = json.loads(message)
                self.exploration_complete = exploration_data.get("complete", False)
                self.exploration_info = exploration_data.get("info", "")

                if self.exploration_complete:
                    logging.info(
                        "Exploration Status: Completed, Info: %s", self.exploration_info
                    )

                    # Trigger the context to stop SLAM
                    self.context_provider.update_context(self.context_aware_text)

                self._monitor.heartbeat("UnitreeGo2FrontierExplorationProvider")

            except json.JSONDecodeError as e:
                logging.error(f"Error decoding exploration message JSON: {e}")
                self._monitor.report_error(
                    "UnitreeGo2FrontierExplorationProvider", str(e)
                )

    def start(self, message_callback: Optional[Callable] = None):
        """
        Start the frontier exploration provider by registering the message callback.
        """
        if not self.running:
            self.register_message_callback(self.frontier_exploration_message_callback)
            self.running = True
            logging.info("Frontier Exploration Provider started")
        else:
            logging.warning("Frontier Exploration Provider is already running")

    @property
    def status(self) -> bool:
        """
        Get the current exploration status.

        Returns
        -------
        bool
            The current exploration status.
        """
        return self.exploration_complete

    @property
    def info(self) -> Optional[str]:
        """
        Get the current exploration info.

        Returns
        -------
        Optional[str]
            The current exploration info.
        """
        return self.exploration_info

    def _recover(self) -> bool:
        """
        Attempt to recover the Frontier exploration provider.

        Returns
        -------
        bool
            True if recovery was successful, False otherwise.
        """
        try:
            logging.info(
                "UnitreeGo2FrontierExplorationProvider: Attempting recovery..."
            )
            self.stop()
            self.start()
            logging.info("UnitreeGo2FrontierExplorationProvider: Recovery successful")
            return True
        except Exception as e:
            logging.error(
                f"UnitreeGo2FrontierExplorationProvider: Recovery failed: {e}"
            )
            return False
