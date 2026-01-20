from unittest.mock import MagicMock, Mock

import pytest

from providers.gallery_identities_provider import (
    GalleryIdentitiesProvider,
    IdentitiesSnapshot,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    GalleryIdentitiesProvider.reset()  # type: ignore
    yield

    try:
        provider = GalleryIdentitiesProvider()
        provider.stop(wait=True)
    except Exception:
        pass

    GalleryIdentitiesProvider.reset()  # type: ignore


def test_identities_snapshot_to_text():
    """Test to_text method of IdentitiesSnapshot."""
    snapshot = IdentitiesSnapshot(
        ts=1234567890.0, total=3, names=["Alice", "Bob", "Charlie"], raw={}
    )

    text = snapshot.to_text()

    assert text == "total=3 ids=[Alice, Bob, Charlie]"


def test_identities_snapshot_to_text_with_duplicates():
    """Test to_text method with duplicate identities."""
    snapshot = IdentitiesSnapshot(
        ts=1234567890.0, total=3, names=["Alice", "Bob", "Alice"], raw={}
    )

    text = snapshot.to_text()

    assert text == "total=3 ids=[Alice, Bob]"


def test_identities_snapshot_to_text_empty():
    """Test to_text method with empty identities."""
    snapshot = IdentitiesSnapshot(ts=1234567890.0, total=0, names=[], raw={})

    text = snapshot.to_text()

    assert text == "total=0 ids=[]"


def test_initialization():
    """Test initialization of the provider."""
    provider = GalleryIdentitiesProvider()

    assert provider is not None
    assert not provider._stop.is_set()


def test_singleton_pattern():
    """Test singleton pattern of the provider."""
    provider1 = GalleryIdentitiesProvider()
    provider2 = GalleryIdentitiesProvider()
    assert provider1 is provider2


def test_start():
    """Test starting the provider."""
    provider = GalleryIdentitiesProvider()
    provider.start()

    assert not provider._stop.is_set()
    assert provider._thread is not None

    provider.stop(wait=True)


def test_start_already_running():
    """Test starting the provider when it's already running."""
    provider = GalleryIdentitiesProvider()
    provider.start()

    thread1 = provider._thread

    provider.start()

    assert provider._thread == thread1

    provider.stop(wait=True)


def test_stop():
    """Test stopping the provider."""
    provider = GalleryIdentitiesProvider()
    provider.start()
    provider.stop(wait=True)

    assert provider._stop.is_set()


def test_register_callback():
    """Test registering a callback."""
    provider = GalleryIdentitiesProvider()
    callback = Mock()

    provider.register_message_callback(callback)

    assert callback in provider._callbacks


def test_unregister_callback():
    """Test unregistering a callback."""
    provider = GalleryIdentitiesProvider()
    callback = Mock()

    provider.register_message_callback(callback)
    provider.unregister_message_callback(callback)

    assert callback not in provider._callbacks


def test_fetch_identities_success():
    """Test successful fetch of identities."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "ok": True,
        "total": 2,
        "identities": [{"id": "Alice"}, {"id": "Bob"}],
    }

    provider = GalleryIdentitiesProvider()
    provider._session.post = MagicMock(return_value=mock_response)

    snapshot = provider._fetch_snapshot()

    assert snapshot is not None
    assert snapshot.total == 2
    assert "Alice" in snapshot.names
    assert "Bob" in snapshot.names


def test_fetch_identities_failure():
    """Test handling of failed fetch due to non-200 response."""
    provider = GalleryIdentitiesProvider()
    provider._session.post = MagicMock(side_effect=Exception("Network error"))

    with pytest.raises(Exception):
        provider._fetch_snapshot()
