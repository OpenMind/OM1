import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np


class MockLidarProvider:
    """
    Singleton class to provide mock lidar scan data to any RPLidar implementation.

    This class serves as a central repository for test lidar scan data that can be
    used by different RPLidar implementations during testing.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MockLidarProvider, cls).__new__(cls)
            cls._instance.test_scans = []
            cls._instance.current_index = 0
            logging.info("Initialized MockLidarProvider singleton")
        return cls._instance

    def load_scans(self, scans: List[np.ndarray]):
        """
        Load a sequence of test lidar scans.

        Parameters
        ----------
        scans : List[np.ndarray]
            List of numpy arrays containing [angle, distance] pairs.
            Distances should be in meters, angles in degrees (0-360).
        """
        self.test_scans = scans
        self.current_index = 0
        logging.info(f"MockLidarProvider loaded {len(self.test_scans)} test scans")

    def load_scans_from_json_files(self, file_paths: List[str], base_dir: Path):
        """
        Load lidar scan data from JSON files.

        Parameters
        ----------
        file_paths : List[str]
            List of file paths to JSON files containing scan data
        base_dir : Path
            Base directory for resolving relative paths
        """
        scans = []

        for file_path in file_paths:
            # Handle both relative and absolute paths
            lidar_file_path = Path(file_path)
            if not lidar_file_path.is_absolute():
                lidar_file_path = base_dir / file_path

            if not lidar_file_path.exists():
                logging.warning(f"Lidar data file not found: {lidar_file_path}")
                continue

            try:
                with open(lidar_file_path, "r") as f:
                    data = json.load(f)

                scan_data = data.get("scan_data", [])
                if scan_data:
                    # Convert to numpy array with distances in meters
                    scan_array = np.array(scan_data)
                    # Convert distances from millimeters to meters
                    scan_array[:, 1] = scan_array[:, 1] / 1000.0

                    scans.append(scan_array)
                    logging.info(
                        f"MockLidarProvider: Loaded {len(scan_data)} scan points from {lidar_file_path}"
                    )
                else:
                    logging.warning(f"MockLidarProvider: No scan_data found in {lidar_file_path}")

            except FileNotFoundError:
                logging.error(f"MockLidarProvider: File not found: {lidar_file_path}")
            except json.JSONDecodeError as e:
                logging.error(f"MockLidarProvider: Invalid JSON in {lidar_file_path}: {e}")
            except Exception as e:
                logging.error(f"MockLidarProvider: Error loading {lidar_file_path}: {e}")

        self.load_scans(scans)

    def get_next_scan(self) -> Optional[np.ndarray]:
        """
        Get the next lidar scan in the sequence.

        Returns
        -------
        Optional[np.ndarray]
            Next test scan as numpy array with [angle, distance] pairs,
            or None if no more scans available
        """
        if not self.test_scans or self.current_index >= len(self.test_scans):
            return None

        scan = self.test_scans[self.current_index]
        self.current_index += 1
        return scan

    def peek_next_scan(self) -> Optional[np.ndarray]:
        """
        Peek at the next lidar scan without advancing the index.

        Returns
        -------
        Optional[np.ndarray]
            Next test scan as numpy array, or None if no more scans available
        """
        if not self.test_scans or self.current_index >= len(self.test_scans):
            return None

        return self.test_scans[self.current_index]

    def has_more_scans(self) -> bool:
        """
        Check if there are more scans available.

        Returns
        -------
        bool
            True if more scans are available, False otherwise
        """
        return bool(self.test_scans and self.current_index < len(self.test_scans))

    def reset(self):
        """Reset the lidar provider to start from the first scan again."""
        self.current_index = 0
        logging.info("MockLidarProvider reset to first scan")

    def clear(self):
        """Clear all loaded scan data."""
        self.test_scans = []
        self.current_index = 0
        logging.info("MockLidarProvider cleared all scan data")

    @property
    def scan_count(self) -> int:
        """Get the total number of loaded scans."""
        return len(self.test_scans)

    @property
    def remaining_scans(self) -> int:
        """Get the number of remaining scans."""
        return max(0, len(self.test_scans) - self.current_index)


# Helper functions to access the singleton
def get_lidar_provider() -> MockLidarProvider:
    """Get the singleton lidar provider instance."""
    return MockLidarProvider()


def load_test_scans(scans: List[np.ndarray]):
    """Load test lidar scans into the provider."""
    provider = get_lidar_provider()
    provider.load_scans(scans)


def load_test_scans_from_files(file_paths: List[str], base_dir: Path):
    """Load test lidar scans from JSON files into the provider."""
    provider = get_lidar_provider()
    provider.load_scans_from_json_files(file_paths, base_dir)


def get_next_lidar_scan() -> Optional[np.ndarray]:
    """Get the next test lidar scan as a numpy array."""
    provider = get_lidar_provider()
    return provider.get_next_scan()


def reset_lidar_provider():
    """Reset the lidar provider to start from the first scan."""
    provider = get_lidar_provider()
    provider.reset()


def clear_lidar_provider():
    """Clear all lidar scan data from the provider."""
    provider = get_lidar_provider()
    provider.clear()
