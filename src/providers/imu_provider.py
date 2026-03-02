import logging
import threading
import time

from .singleton import singleton


@singleton
class IMUProvider:
    """
    Singleton provider for IMU (Inertial Measurement Unit) data.

    Stores and distributes accelerometer, gyroscope, and orientation
    data to all OM1 components. Also handles fall/impact detection.
    """

    def __init__(self):
        """Initialize the IMUProvider."""
        logging.info("Booting IMUProvider")

        self._lock = threading.Lock()

        # Accelerometer (m/s^2)
        self.accel_x: float = 0.0
        self.accel_y: float = 0.0
        self.accel_z: float = 0.0

        # Gyroscope (deg/s)
        self.gyro_x: float = 0.0
        self.gyro_y: float = 0.0
        self.gyro_z: float = 0.0

        # Orientation (degrees)
        self.roll: float = 0.0
        self.pitch: float = 0.0
        self.yaw: float = 0.0

        # Fall/impact detection
        self.is_fallen: bool = False
        self.impact_detected: bool = False

        # Thresholds
        self.fall_threshold: float = 45.0
        self.impact_threshold: float = 20.0

        # Timestamps
        self.last_update: float = 0.0

    def update(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> None:
        """
        Update IMU data and evaluate fall/impact detection.

        Parameters
        ----------
        accel_x, accel_y, accel_z : float
            Accelerometer readings in m/s^2.
        gyro_x, gyro_y, gyro_z : float
            Gyroscope readings in deg/s.
        roll, pitch, yaw : float
            Orientation angles in degrees.
        """
        with self._lock:
            self.accel_x = accel_x
            self.accel_y = accel_y
            self.accel_z = accel_z

            self.gyro_x = gyro_x
            self.gyro_y = gyro_y
            self.gyro_z = gyro_z

            self.roll = roll
            self.pitch = pitch
            self.yaw = yaw

            self.last_update = time.time()

            # Fall detection: robot tilted beyond threshold
            self.is_fallen = (
                abs(self.roll) > self.fall_threshold
                or abs(self.pitch) > self.fall_threshold
            )

            # Impact detection: sudden acceleration spike
            accel_magnitude = (accel_x**2 + accel_y**2 + accel_z**2) ** 0.5
            self.impact_detected = accel_magnitude > self.impact_threshold

            if self.is_fallen:
                logging.warning(
                    f"IMUProvider: fall detected - roll={roll:.1f} pitch={pitch:.1f}"
                )
            if self.impact_detected:
                logging.warning(
                    f"IMUProvider: impact detected - magnitude={accel_magnitude:.2f}"
                )

    @property
    def state(self) -> dict:
        """
        Get current IMU state as dictionary.

        Returns
        -------
        dict
            Current IMU readings and detection flags.
        """
        with self._lock:
            return {
                "accel_x": self.accel_x,
                "accel_y": self.accel_y,
                "accel_z": self.accel_z,
                "gyro_x": self.gyro_x,
                "gyro_y": self.gyro_y,
                "gyro_z": self.gyro_z,
                "roll": self.roll,
                "pitch": self.pitch,
                "yaw": self.yaw,
                "is_fallen": self.is_fallen,
                "impact_detected": self.impact_detected,
                "last_update": self.last_update,
            }

    def reset_alerts(self) -> None:
        """Reset fall and impact detection flags."""
        with self._lock:
            self.is_fallen = False
            self.impact_detected = False
            logging.info("IMUProvider: alerts reset")

    def stop(self) -> None:
        """Stop the IMUProvider and clean up resources."""
        logging.info("IMUProvider stopped")
