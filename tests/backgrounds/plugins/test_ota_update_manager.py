"""Unit tests for OTAUpdateManager background plugin."""

import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

_mock_zenoh = types.ModuleType("zenoh")
_mock_zenoh.Config = object  # type: ignore[attr-defined]
_mock_zenoh.Session = object  # type: ignore[attr-defined]
sys.modules.setdefault("zenoh", _mock_zenoh)

from backgrounds.plugins.ota_update_manager import (  # noqa: E402
    OTAUpdateManager,
    OTAUpdateManagerConfig,
)


@pytest.fixture
def config() -> OTAUpdateManagerConfig:
    return OTAUpdateManagerConfig(
        check_interval=1,
        auto_update=False,
        update_url="http://mock-server",
        require_balance_check=True,
    )


@pytest.fixture
def mock_provider() -> MagicMock:
    """A fully-configured mock OTAProvider."""
    p = MagicMock()
    p.base_url = "http://mock-server"
    p.download_path = MagicMock()
    p.download_path.__truediv__ = lambda self, other: MagicMock()
    return p


@pytest.fixture
def manager(config, mock_provider) -> OTAUpdateManager:
    with patch(
        "backgrounds.plugins.ota_update_manager.OTAProvider",
        return_value=mock_provider,
    ):
        mgr = OTAUpdateManager(config)
    mgr.ota_provider = mock_provider
    return mgr


def _update_info(
    version: str = "2.0.0",
    price: float = 10.0,
    package_url: str = "http://example.com/update.zip",
    sha256: str = "abc123",
) -> dict:
    info: dict = {
        "version": version,
        "price": price,
        "package_url": package_url,
    }
    if sha256:
        info["sha256"] = sha256
    return info


class TestInit:
    def test_attributes(self, manager, config):
        assert manager.check_interval == 1
        assert manager.auto_update is False
        assert manager.require_balance is True
        assert isinstance(manager._ota_stop_event, threading.Event)

    def test_base_url_overridden(self, manager):
        assert manager.ota_provider.base_url == "http://mock-server"

    def test_no_url_override_keeps_provider_default(self):
        cfg = OTAUpdateManagerConfig(check_interval=60)
        with patch("backgrounds.plugins.ota_update_manager.OTAProvider") as MockProv:
            mock = MagicMock()
            mock.base_url = "http://original"
            MockProv.return_value = mock
            mgr = OTAUpdateManager(cfg)
        # base_url should NOT be overridden
        assert mgr.ota_provider.base_url == "http://original"


class TestLifecycle:
    def test_run_calls_check_and_stop_exits(self, manager):
        manager._check_and_update = MagicMock()
        t = threading.Thread(target=manager.run)
        t.start()
        time.sleep(0.05)
        manager.stop()
        t.join(timeout=2)
        assert not t.is_alive()
        assert manager._check_and_update.called

    def test_stop_sets_event(self, manager):
        manager.stop()
        assert manager._ota_stop_event.is_set()

    def test_exception_in_cycle_does_not_crash_thread(self, manager):
        manager._check_and_update = MagicMock(side_effect=RuntimeError("boom"))
        t = threading.Thread(target=manager.run)
        t.start()
        time.sleep(0.05)
        manager.stop()
        t.join(timeout=2)
        assert not t.is_alive()


class TestNoUpdate:
    def test_no_update_returns_early(self, manager):
        manager.ota_provider.check_for_updates.return_value = (False, None)
        manager._check_and_update()
        manager.ota_provider.get_balance.assert_not_called()
        manager.ota_provider.download_update.assert_not_called()


class TestBalanceCheck:
    def test_sufficient_balance_proceeds_to_download(self, manager):
        manager.ota_provider.check_for_updates.return_value = (True, _update_info())
        manager.ota_provider.get_balance.return_value = 100.0
        manager.ota_provider.download_update.return_value = True
        manager.ota_provider.verify_package.return_value = True
        manager.ota_provider.apply_update.return_value = True
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.get_balance.assert_called_once()
        manager.ota_provider.download_update.assert_called_once_with(
            "2.0.0", "http://example.com/update.zip"
        )

    def test_insufficient_balance_stops_pipeline(self, manager):
        manager.ota_provider.check_for_updates.return_value = (
            True,
            _update_info(price=50.0),
        )
        manager.ota_provider.get_balance.return_value = 5.0
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.download_update.assert_not_called()
        manager.ota_provider.apply_update.assert_not_called()

    def test_none_balance_stops_pipeline(self, manager):
        manager.ota_provider.check_for_updates.return_value = (True, _update_info())
        manager.ota_provider.get_balance.return_value = None
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.download_update.assert_not_called()

    def test_free_update_skips_balance_check(self, manager):
        manager.ota_provider.check_for_updates.return_value = (
            True,
            _update_info(price=0.0),
        )
        manager.ota_provider.download_update.return_value = True
        manager.ota_provider.verify_package.return_value = True
        manager.ota_provider.apply_update.return_value = True
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.get_balance.assert_not_called()
        manager.ota_provider.download_update.assert_called_once()

    def test_balance_check_skipped_when_disabled(self, manager):
        manager.require_balance = False
        manager.ota_provider.check_for_updates.return_value = (True, _update_info())
        manager.ota_provider.download_update.return_value = True
        manager.ota_provider.verify_package.return_value = True
        manager.ota_provider.apply_update.return_value = True
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.get_balance.assert_not_called()


class TestDownloadAndVerify:
    def test_missing_package_url_stops_pipeline(self, manager):
        info = {"version": "2.0.0", "price": 0}
        manager.ota_provider.check_for_updates.return_value = (True, info)
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.download_update.assert_not_called()

    def test_download_failure_stops_pipeline(self, manager):
        manager.ota_provider.check_for_updates.return_value = (
            True,
            _update_info(price=0.0),
        )
        manager.ota_provider.download_update.return_value = False
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.verify_package.assert_not_called()
        manager.ota_provider.apply_update.assert_not_called()

    def test_verify_failure_stops_pipeline(self, manager):
        manager.ota_provider.check_for_updates.return_value = (
            True,
            _update_info(price=0.0),
        )
        manager.ota_provider.download_update.return_value = True
        manager.ota_provider.verify_package.return_value = False
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.apply_update.assert_not_called()

    def test_no_sha256_skips_verify(self, manager):
        info = _update_info(price=0.0)
        del info["sha256"]
        manager.ota_provider.check_for_updates.return_value = (True, info)
        manager.ota_provider.download_update.return_value = True
        manager.ota_provider.apply_update.return_value = True
        manager.auto_update = True

        manager._check_and_update()

        manager.ota_provider.verify_package.assert_not_called()
        manager.ota_provider.apply_update.assert_called_once()


class TestApplyAndRollback:
    def _setup_happy_path(self, manager, price: float = 10.0):
        manager.ota_provider.check_for_updates.return_value = (
            True,
            _update_info(price=price),
        )
        manager.ota_provider.get_balance.return_value = 100.0
        manager.ota_provider.download_update.return_value = True
        manager.ota_provider.verify_package.return_value = True
        manager.auto_update = True

    def test_successful_paid_update_records_transaction(self, manager):
        self._setup_happy_path(manager, price=10.0)
        manager.ota_provider.apply_update.return_value = True

        manager._check_and_update()

        manager.ota_provider.apply_update.assert_called_once_with("2.0.0")
        manager.ota_provider.record_transaction.assert_called_once_with(
            10.0, "OM1 update to version 2.0.0"
        )

    def test_successful_free_update_no_transaction(self, manager):
        self._setup_happy_path(manager, price=0.0)
        manager.ota_provider.apply_update.return_value = True

        manager._check_and_update()

        manager.ota_provider.record_transaction.assert_not_called()

    def test_apply_failure_triggers_rollback(self, manager):
        self._setup_happy_path(manager)
        manager.ota_provider.apply_update.return_value = False

        manager._check_and_update()

        manager.ota_provider.rollback.assert_called_once()
        manager.ota_provider.record_transaction.assert_not_called()

    def test_auto_update_false_logs_ready(self, manager):
        manager.ota_provider.check_for_updates.return_value = (
            True,
            _update_info(price=0.0),
        )
        manager.ota_provider.download_update.return_value = True
        manager.ota_provider.verify_package.return_value = True
        manager.auto_update = False

        with patch.object(manager.logger, "info") as mock_log:
            manager._check_and_update()

        logged_messages = [str(c) for c in mock_log.call_args_list]
        assert any("auto-update disabled" in m for m in logged_messages)
        manager.ota_provider.apply_update.assert_not_called()
