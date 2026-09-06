import logging
import threading

from backgrounds.base import Background, BackgroundConfig
from providers.context_provider import ContextProvider
from providers.imu_provider import IMUProvider


class IMUFallDetector(Background[BackgroundConfig]):
    """
    Background task that continuously monitors IMU data for fall
    and impact events, updating the context provider when detected.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize the IMUFallDetector background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration for the background task.
        """
        super().__init__(config)

        self._lock = threading.Lock()
        self.imu_provider = IMUProvider()
        self.context_provider = ContextProvider()

        self._fall_reported: bool = False
        self._impact_reported: bool = False

        logging.info("IMUFallDetector background task initialized.")

    def run(self) -> None:
        """
        Monitor IMU state and update context on fall or impact detection.
        """
        state = self.imu_provider.state

        with self._lock:
            if state["is_fallen"] and not self._fall_reported:
                logging.warning("IMUFallDetector: fall detected, updating context.")
                self.context_provider.update_context(
                    {
                        "imu_fall_detected": True,
                        "imu_roll": state["roll"],
                        "imu_pitch": state["pitch"],
                    }
                )
                self._fall_reported = True

            elif not state["is_fallen"] and self._fall_reported:
                logging.info("IMUFallDetector: fall resolved, resetting context.")
                self.context_provider.update_context({"imu_fall_detected": False})
                self._fall_reported = False

            if state["impact_detected"] and not self._impact_reported:
                logging.warning("IMUFallDetector: impact detected, updating context.")
                self.context_provider.update_context({"imu_impact_detected": True})
                self._impact_reported = True

            elif not state["impact_detected"] and self._impact_reported:
                self.context_provider.update_context({"imu_impact_detected": False})
                self._impact_reported = False

        self.sleep(0.1)

    def stop(self) -> None:
        """
        Stop the IMUFallDetector background task.
        """
        logging.info("Stopping IMUFallDetector background task.")
