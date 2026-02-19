"""
OTA Provider for managing over-the-air updates.

This provider handles checking for updates, downloading packages,
verifying integrity, and applying updates. It integrates with OpenMind's
OMCU token system for paid updates.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from .singleton import singleton


@singleton
class OTAProvider:
    """
    Provider for OTA update management.

    This is a singleton that manages the update process, including
    checking for new versions, downloading, verification, and installation.
    Replace mock endpoints with real server URLs when available.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = os.environ.get("OTA_UPDATE_URL", "http://localhost:8000")
        self.api_key: Optional[str] = None
        self.current_version = self._get_current_version()
        self.update_available = False
        self.update_info: Dict = {}
        self.download_path = Path("/tmp/om1_update")

        self.logger.info(
            "OTAProvider initialized. Current version: %s", self.current_version
        )

    def set_api_key(self, api_key: str) -> None:
        """Set the API key for authentication."""
        self.api_key = api_key

    def _get_current_version(self) -> str:
        """Read current version from version.py or VERSION file."""
        try:
            from runtime.version import __version__  # type: ignore[import]

            return __version__  # pragma: no cover
        except ImportError:
            version_file = Path(__file__).parent.parent / "runtime" / "version.py"
            if version_file.exists():
                with open(version_file, "r") as f:
                    for line in f:
                        if line.startswith("__version__"):
                            return line.split("=")[1].strip().strip("\"'")
            return "0.0.0"

    def _get_headers(self) -> Dict[str, str]:
        """Prepare headers for HTTP requests."""
        headers: Dict[str, str] = {}
        if self.api_key:
            # JWT token from Clerk — see docs/api-reference/endpoints/account_and_key_management.md
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def check_for_updates(self) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a new version is available.

        Returns
        -------
        Tuple[bool, Optional[Dict]]
            (update_available, update_info)
        """
        try:
            url = f"{self.base_url}/api/updates/latest"
            params = {"current_version": self.current_version}
            resp = requests.get(
                url, params=params, headers=self._get_headers(), timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("version") != self.current_version:
                    self.update_available = True
                    self.update_info = data
                    self.logger.info("Update available: %s", data["version"])
                    return True, data
                self.logger.debug("No update available")
                return False, None
            self.logger.warning("Update check failed: %s", resp.status_code)
            return False, None
        except Exception as e:
            self.logger.error("Error checking for updates: %s", e)
            return False, None

    def get_balance(self) -> Optional[float]:
        """
        Get OMCU balance from OpenMind API.

        Returns
        -------
        Optional[float]
            Balance in OMCU, or None if failed.
        """
        try:
            url = f"{self.base_url}/api/account/balance"
            resp = requests.get(url, headers=self._get_headers(), timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                balance = float(data.get("omcu_balance", 0))
                self.logger.info("Current OMCU balance: %s", balance)
                return balance
            self.logger.warning("Balance check failed: %s", resp.status_code)
            return None
        except Exception as e:
            self.logger.error("Error getting balance: %s", e)
            return None

    def download_update(self, version: str, package_url: str) -> bool:
        """
        Download update package.

        Parameters
        ----------
        version : str
            Version string.
        package_url : str
            URL to download the package from.

        Returns
        -------
        bool
            True if download successful.
        """
        try:
            self.download_path.mkdir(parents=True, exist_ok=True)
            local_file = self.download_path / f"om1_update_{version}.zip"
            resp = requests.get(package_url, stream=True, timeout=30)
            if resp.status_code == 200:
                with open(local_file, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                self.logger.info("Downloaded update to %s", local_file)
                return True
            self.logger.error("Download failed: %s", resp.status_code)
            return False
        except Exception as e:
            self.logger.error("Error downloading update: %s", e)
            return False

    def verify_package(self, file_path: Path, expected_sha256: str) -> bool:
        """
        Verify package integrity using SHA-256.

        Parameters
        ----------
        file_path : Path
            Path to the downloaded file.
        expected_sha256 : str
            Expected SHA-256 hash.

        Returns
        -------
        bool
            True if hash matches.
        """
        try:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            if actual_hash == expected_sha256:
                self.logger.info("Package verified successfully")
                return True
            self.logger.error(
                "Hash mismatch: expected %s, got %s", expected_sha256, actual_hash
            )
            return False
        except Exception as e:
            self.logger.error("Error verifying package: %s", e)
            return False

    def apply_update(self, version: str) -> bool:
        """
        Apply the downloaded update.

        Parameters
        ----------
        version : str
            Version being installed.

        Returns
        -------
        bool
            True if update successful.

        Notes
        -----
        Currently a stub — replace with real install logic (backup,
        replace files, trigger restart).
        """
        try:
            self.logger.info("Applying update to version %s...", version)
            time.sleep(2)  # Simulate installation
            # See docs/api-reference/endpoints/ota_update.md - Section "apply_update()"
            self.logger.info("Update applied successfully")
            return True
        except Exception as e:
            self.logger.error("Error applying update: %s", e)
            return False

    def record_transaction(self, amount: float, description: str) -> bool:
        """
        Record a debit transaction after a successful paid update.

        Parameters
        ----------
        amount : float
            Amount of OMCU to debit.
        description : str
            Human-readable description for the transaction.

        Returns
        -------
        bool
            True if the transaction was recorded successfully.
        """
        try:
            url = f"{self.base_url}/api/transactions"
            payload = {
                "amount": amount,
                "description": description,
                "timestamp": time.time(),
            }
            resp = requests.post(
                url, json=payload, headers=self._get_headers(), timeout=10
            )
            if resp.status_code in (200, 201):
                self.logger.info(
                    "Transaction recorded: %s OMCU for %s", amount, description
                )
                return True
            self.logger.warning("Transaction failed: %s", resp.status_code)
            return False
        except Exception as e:
            self.logger.error("Error recording transaction: %s", e)
            return False

    def rollback(self) -> bool:
        """
        Rollback to the previous version if an update fails.

        Returns
        -------
        bool
            True if rollback successful.

        Notes
        -----
        Currently a stub — replace with real rollback logic.
        """
        try:
            self.logger.info("Rolling back to previous version...")
            time.sleep(1)  # Simulate rollback
            # See docs/api-reference/endpoints/ota_update.md - Section "rollback()"
            self.logger.info("Rollback completed")
            return True
        except Exception as e:
            self.logger.error("Error during rollback: %s", e)
            return False
