"""Unit tests for OTAProvider."""

import hashlib
import sys
import types
from unittest.mock import MagicMock, mock_open, patch

import pytest

_mock_zenoh = types.ModuleType("zenoh")
_mock_zenoh.Config = object  # type: ignore[attr-defined]
_mock_zenoh.Session = object  # type: ignore[attr-defined]
sys.modules.setdefault("zenoh", _mock_zenoh)

from providers.ota_provider import OTAProvider  # noqa: E402


def _fresh_provider() -> OTAProvider:
    """Return a new OTAProvider with the singleton reset."""
    OTAProvider._singleton_class._singleton_instance = None  # type: ignore[attr-defined]
    p = OTAProvider()
    p.api_key = None
    return p


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure each test starts with a fresh OTAProvider singleton."""
    OTAProvider._singleton_class._singleton_instance = None  # type: ignore[attr-defined]
    yield
    OTAProvider._singleton_class._singleton_instance = None  # type: ignore[attr-defined]


@pytest.fixture
def provider() -> OTAProvider:
    return _fresh_provider()


class TestSingleton:
    def test_singleton_returns_same_instance(self):
        p1 = _fresh_provider()
        p2 = OTAProvider()
        assert p1 is p2


class TestGetCurrentVersion:
    def test_returns_fallback_when_import_and_file_missing(self):
        OTAProvider._singleton_class._singleton_instance = None  # type: ignore[attr-defined]
        with patch("pathlib.Path.exists", return_value=False):
            p = OTAProvider()
        assert p.current_version == "0.0.0"

    def test_reads_version_from_file(self, tmp_path):
        version_file = tmp_path / "version.py"
        version_file.write_text('__version__ = "3.1.4"\n')
        OTAProvider._singleton_class._singleton_instance = None  # type: ignore[attr-defined]
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data='__version__ = "3.1.4"\n')),
        ):
            p = OTAProvider()
            version = p._get_current_version()
        assert version == "3.1.4"


class TestHeaders:
    def test_no_api_key_returns_empty_headers(self, provider):
        assert provider._get_headers() == {}

    def test_with_api_key_includes_bearer(self, provider):
        provider.api_key = "secret-key"
        assert provider._get_headers() == {"Authorization": "Bearer secret-key"}

    def test_set_api_key(self, provider):
        provider.set_api_key("new-key")
        assert provider.api_key == "new-key"


class TestCheckForUpdates:
    @patch("requests.get")
    def test_update_available(self, mock_get, provider):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"version": "2.0.0", "price": 10},
        )
        provider.current_version = "1.0.0"
        available, info = provider.check_for_updates()
        assert available is True
        assert info is not None
        assert info["version"] == "2.0.0"

    @patch("requests.get")
    def test_no_update_same_version(self, mock_get, provider):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"version": "1.0.0"},
        )
        provider.current_version = "1.0.0"
        available, info = provider.check_for_updates()
        assert available is False
        assert info is None

    @patch("requests.get")
    def test_http_error_returns_false(self, mock_get, provider):
        mock_get.return_value = MagicMock(status_code=503)
        available, info = provider.check_for_updates()
        assert available is False
        assert info is None

    def test_network_exception_returns_false(self, provider):
        with patch("requests.get", side_effect=ConnectionError("timeout")):
            available, info = provider.check_for_updates()
        assert available is False
        assert info is None


class TestGetBalance:
    @patch("requests.get")
    def test_success(self, mock_get, provider):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"omcu_balance": 250},
        )
        assert provider.get_balance() == 250.0

    @patch("requests.get")
    def test_http_error_returns_none(self, mock_get, provider):
        mock_get.return_value = MagicMock(status_code=401)
        assert provider.get_balance() is None

    def test_network_exception_returns_none(self, provider):
        with patch("requests.get", side_effect=ConnectionError("timeout")):
            assert provider.get_balance() is None


class TestDownloadUpdate:
    @patch("requests.get")
    def test_success(self, mock_get, provider):
        mock_get.return_value = MagicMock(
            status_code=200,
            iter_content=lambda chunk_size: [b"data"],
        )
        with patch("pathlib.Path.mkdir"), patch("builtins.open", mock_open()):
            assert (
                provider.download_update("1.0.0", "http://example.com/pkg.zip") is True
            )

    @patch("requests.get")
    def test_http_error_returns_false(self, mock_get, provider):
        mock_get.return_value = MagicMock(status_code=404)
        assert provider.download_update("1.0.0", "http://example.com/pkg.zip") is False

    def test_network_exception_returns_false(self, provider):
        with patch("requests.get", side_effect=ConnectionError("timeout")):
            assert provider.download_update("1.0.0", "http://example.com") is False


class TestVerifyPackage:
    def test_correct_hash(self, provider, tmp_path):
        content = b"hello om1"
        f = tmp_path / "pkg.zip"
        f.write_bytes(content)
        assert provider.verify_package(f, hashlib.sha256(content).hexdigest()) is True

    def test_wrong_hash(self, provider, tmp_path):
        f = tmp_path / "pkg.zip"
        f.write_bytes(b"hello om1")
        assert provider.verify_package(f, "deadbeef") is False

    def test_missing_file(self, provider, tmp_path):
        assert provider.verify_package(tmp_path / "ghost.zip", "abc") is False


class TestApplyUpdate:
    @patch("time.sleep")
    def test_success(self, mock_sleep, provider):
        assert provider.apply_update("2.0.0") is True
        mock_sleep.assert_called_once_with(2)

    def test_exception_returns_false(self, provider):
        with patch("time.sleep", side_effect=RuntimeError("disk full")):
            assert provider.apply_update("2.0.0") is False


class TestRecordTransaction:
    @patch("requests.post")
    def test_success_201(self, mock_post, provider):
        mock_post.return_value = MagicMock(status_code=201)
        assert provider.record_transaction(10.0, "Update to 2.0.0") is True

    @patch("requests.post")
    def test_success_200(self, mock_post, provider):
        mock_post.return_value = MagicMock(status_code=200)
        assert provider.record_transaction(5.0, "Test") is True

    @patch("requests.post")
    def test_http_error_returns_false(self, mock_post, provider):
        mock_post.return_value = MagicMock(status_code=400)
        assert provider.record_transaction(10.0, "Test") is False

    def test_network_exception_returns_false(self, provider):
        with patch("requests.post", side_effect=ConnectionError("timeout")):
            assert provider.record_transaction(10.0, "Test") is False


class TestRollback:
    @patch("time.sleep")
    def test_success(self, mock_sleep, provider):
        assert provider.rollback() is True
        mock_sleep.assert_called_once_with(1)

    def test_exception_returns_false(self, provider):
        with patch("time.sleep", side_effect=RuntimeError("fs error")):
            assert provider.rollback() is False
