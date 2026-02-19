"""
OTA Update Manager background plugin.

Periodically checks for updates and applies them if available.
"""

import logging
import threading
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.ota_provider import OTAProvider


class OTAUpdateManagerConfig(BackgroundConfig):
    """
    Configuration for OTA Update Manager.

    Attributes
    ----------
    check_interval : int
        Seconds between update checks (default: 3600 = 1 hour).
    auto_update : bool
        Whether to apply updates automatically (default: False).
    update_url : Optional[str]
        Base URL for update server; overrides the OTA_UPDATE_URL env var.
    require_balance_check : bool
        Check OMCU balance before applying paid updates (default: True).
    """

    check_interval: int = Field(
        default=3600, gt=0, description="Seconds between update checks"
    )
    auto_update: bool = Field(default=False, description="Apply updates automatically")
    update_url: Optional[str] = Field(
        default=None, description="Base URL for update server"
    )
    require_balance_check: bool = Field(
        default=True, description="Check OMCU balance before paid updates"
    )


class OTAUpdateManager(Background[OTAUpdateManagerConfig]):
    """
    Background task for managing over-the-air updates.

    Periodically checks for new versions of OM1. If an update is
    available, it can apply it automatically (when ``auto_update=True``)
    or log that the update is ready for manual intervention.
    """

    def __init__(self, config: OTAUpdateManagerConfig):
        super().__init__(config)

        self.ota_provider = OTAProvider()
        self.check_interval = config.check_interval
        self.auto_update = config.auto_update
        self.require_balance = config.require_balance_check

        if config.update_url:
            self.ota_provider.base_url = config.update_url

        self._ota_stop_event: threading.Event = threading.Event()
        self.logger = logging.getLogger(__name__)
        self.logger.info("OTAUpdateManager initialized")

    def run(self) -> None:
        """Main background loop — runs until ``stop()`` is called."""
        self.logger.info("OTAUpdateManager background thread started")
        while not self._ota_stop_event.is_set():
            try:
                self._check_and_update()
            except Exception as e:
                self.logger.error("Unhandled error in OTA update cycle: %s", e)

            # Block until the next interval or until stop() is called.
            self._ota_stop_event.wait(timeout=self.check_interval)

        self.logger.info("OTAUpdateManager background thread stopped")

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self.logger.info("Stopping OTAUpdateManager")
        self._ota_stop_event.set()

    def _check_and_update(self) -> None:
        """Perform one full update-check cycle."""
        # 1. Check for updates
        update_available, update_info = self.ota_provider.check_for_updates()
        if not update_available or not update_info:
            self.logger.debug("No update available")
            return

        version: str = update_info.get("version") or ""
        self.logger.info("Update available: %s", version)

        # 2. Balance check (only for paid updates)
        price: float = float(update_info.get("price", 0))
        if price > 0 and self.require_balance:
            balance = self.ota_provider.get_balance()
            if balance is None or balance < price:
                self.logger.warning(
                    "Insufficient OMCU balance for update %s "
                    "(required: %s, available: %s)",
                    version,
                    price,
                    balance,
                )
                # See docs/api-reference/endpoints/ota_update.md - TTS notification to be implemented
                return
            self.logger.info("Balance sufficient: %s OMCU", balance)

        # 3. Download
        package_url: Optional[str] = update_info.get("package_url")
        if not package_url:
            self.logger.error("No package_url in update info for version %s", version)
            return

        if not self.ota_provider.download_update(version, package_url):
            self.logger.error("Download failed for version %s", version)
            return

        # 4. Verify integrity
        expected_hash: Optional[str] = update_info.get("sha256")
        if expected_hash:
            file_path = self.ota_provider.download_path / f"om1_update_{version}.zip"
            if not self.ota_provider.verify_package(file_path, expected_hash):
                self.logger.error("Package verification failed for version %s", version)
                return

        # 5. Apply (only when auto-update is enabled)
        if self.auto_update:
            self.logger.info("Applying update %s", version)
            if self.ota_provider.apply_update(version):
                if price > 0:
                    self.ota_provider.record_transaction(
                        price, f"OM1 update to version {version}"
                    )
                self.logger.info("Update to %s successful", version)
                # See docs/api-reference/endpoints/ota_update.md - restart or TTS notification to be implemented
            else:
                self.logger.error("Update to %s failed, attempting rollback", version)
                self.ota_provider.rollback()
        else:
            self.logger.info("Update %s ready (auto-update disabled)", version)
            # See docs/api-reference/endpoints/ota_update.md - TTS notification to be implemented that an update is waiting
