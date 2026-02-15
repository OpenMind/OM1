import logging
from typing import Dict, List, Optional


class MockStateProvider:
    """
    Singleton class to provide mock state data for battery, odometry, and GPS inputs.

    This class serves as a central repository for test state data that can be
    used by different state-based input implementations during testing.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MockStateProvider, cls).__new__(cls)
            cls._instance.battery_data = []
            cls._instance.battery_index = 0
            cls._instance.odometry_data = []
            cls._instance.odometry_index = 0
            cls._instance.gps_data = []
            cls._instance.gps_index = 0
            logging.info("Initialized MockStateProvider singleton")
        return cls._instance

    def load_battery_data(self, data: List[List[float]]):
        """
        Load battery state data.

        Parameters
        ----------
        data : List[List[float]]
            List of [percent, voltage, amperes] values
        """
        self.battery_data = data
        self.battery_index = 0
        logging.info(f"MockStateProvider loaded {len(data)} battery data entries")

    def get_next_battery(self) -> Optional[List[float]]:
        """
        Get the next battery state data.

        Returns
        -------
        Optional[List[float]]
            Next battery data [percent, voltage, amperes] or None
        """
        if not self.battery_data or self.battery_index >= len(self.battery_data):
            return None

        data = self.battery_data[self.battery_index]
        self.battery_index += 1
        return data

    def load_odometry_data(self, data: List[Dict]):
        """
        Load odometry state data.

        Parameters
        ----------
        data : List[Dict]
            List of odometry dicts with x, y, yaw, moving, body_attitude keys
        """
        self.odometry_data = data
        self.odometry_index = 0
        logging.info(f"MockStateProvider loaded {len(data)} odometry data entries")

    def get_next_odometry(self) -> Optional[Dict]:
        """
        Get the next odometry state data.

        Returns
        -------
        Optional[Dict]
            Next odometry data dict or None
        """
        if not self.odometry_data or self.odometry_index >= len(self.odometry_data):
            return None

        data = self.odometry_data[self.odometry_index]
        self.odometry_index += 1
        return data

    def load_gps_data(self, data: List[Dict]):
        """
        Load GPS state data.

        Parameters
        ----------
        data : List[Dict]
            List of GPS dicts with gps_lat, gps_lon, gps_alt, gps_qua keys
        """
        self.gps_data = data
        self.gps_index = 0
        logging.info(f"MockStateProvider loaded {len(data)} GPS data entries")

    def get_next_gps(self) -> Optional[Dict]:
        """
        Get the next GPS state data.

        Returns
        -------
        Optional[Dict]
            Next GPS data dict or None
        """
        if not self.gps_data or self.gps_index >= len(self.gps_data):
            return None

        data = self.gps_data[self.gps_index]
        self.gps_index += 1
        return data

    def clear_all(self):
        """Clear all loaded data and reset all indices."""
        self.battery_data = []
        self.battery_index = 0
        self.odometry_data = []
        self.odometry_index = 0
        self.gps_data = []
        self.gps_index = 0


def get_state_provider() -> MockStateProvider:
    """Get the singleton state provider instance."""
    return MockStateProvider()


def get_next_battery() -> Optional[List[float]]:
    """Get the next battery state data."""
    return get_state_provider().get_next_battery()


def get_next_odometry() -> Optional[Dict]:
    """Get the next odometry state data."""
    return get_state_provider().get_next_odometry()


def get_next_gps() -> Optional[Dict]:
    """Get the next GPS state data."""
    return get_state_provider().get_next_gps()


def clear_state_provider():
    """Clear all loaded state data."""
    get_state_provider().clear_all()
