import logging
from typing import Callable, Optional

import zenoh

from zenoh_msgs import ChargingStatus

from .singleton import singleton
from .zenoh_listener_provider import ZenohListenerProvider


@singleton
class UnitreeGo2ChargingProvider(ZenohListenerProvider):
    """
    Charging Status Provider for Unitree Go2 robot.
    Subscribes to om/go2/charging_status and exposes latest status.
    """

    def __init__(self, topic: str = "om/go2/charging_status"):
        """
        Initialize the Charging Provider with a specific topic.

        Parameters
        ----------
        topic : str, optional
            The topic on which to subscribe for charging status messages
            (default is "om/go2/charging_status").
        """
        super().__init__(topic)
        logging.info("Charging Provider initialized with topic: %s", topic)

        self.latest_status: Optional[int] = None
        self.status_history: list = []

    def charging_message_callback(self, data: zenoh.Sample):
        """
        Process an incoming ChargingStatus message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        if data.payload:
            try:
                message: ChargingStatus = ChargingStatus.deserialize(
                    data.payload.to_bytes()
                )
                self.latest_status = message.code
                self.status_history.append(message.code)
                logging.info("Received charging status: %s", message.code)
            except Exception as e:
                logging.error("Failed to parse ChargingStatus: %s", e)
        else:
            logging.warning("Received empty ChargingStatus message")

    def start(self, message_callback: Optional[Callable] = None):
        """
        Start the Charging Provider by registering the message callback.

        Parameters
        ----------
        message_callback : Optional[Callable], optional
        """
        if not self.running:
            self.register_message_callback(self.charging_message_callback)
            self.running = True
            logging.info("Charging Provider started and listening for messages")
        else:
            logging.warning("Charging Provider is already running")

    @property
    def charging_status(self) -> Optional[int]:
        """
        Get the current charging status.

        Returns
        -------
        Optional[int]
            The current charging status if available, None otherwise.
        """
        return self.latest_status

    def get_charging_status(self) -> Optional[int]:
        """
        Get the current charging status (method form for compatibility).

        Returns
        -------
        Optional[int]
            The current charging status if available, None otherwise.
        """
        return self.latest_status

    def get_status_history(self) -> list:
        """
        Get the history of charging statuses.

        Returns
        -------
        list
            List of all charging statuses received.
        """
        return self.status_history
