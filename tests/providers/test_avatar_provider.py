from unittest.mock import MagicMock, patch

import pytest

from providers.avatar_provider import AvatarProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    AvatarProvider.reset()  # type: ignore
    yield

    try:
        provider = AvatarProvider()
        provider.stop()
    except Exception:
        pass

    AvatarProvider.reset()  # type: ignore


@pytest.fixture
def mock_zenoh():
    with patch("providers.avatar_provider.open_zenoh_session") as mock_session:
        mock_session_instance = MagicMock()
        mock_publisher = MagicMock()
        mock_healthcheck_publisher = MagicMock()
        mock_subscriber = MagicMock()
        mock_session_instance.declare_publisher.side_effect = [
            mock_publisher,
            mock_healthcheck_publisher,
        ]
        mock_session_instance.declare_subscriber.return_value = mock_subscriber
        mock_session.return_value = mock_session_instance
        yield (
            mock_session,
            mock_session_instance,
            mock_publisher,
            mock_healthcheck_publisher,
            mock_subscriber,
        )


def test_initialization(mock_zenoh):
    (
        mock_session,
        mock_session_instance,
        mock_publisher,
        mock_healthcheck_publisher,
        mock_subscriber,
    ) = mock_zenoh
    provider = AvatarProvider()

    assert provider.running
    assert provider.session == mock_session_instance
    assert provider.avatar_publisher == mock_publisher
    assert provider.avatar_healthcheck_publisher == mock_healthcheck_publisher
    assert provider.avatar_subscriber == mock_subscriber


def test_singleton_pattern():
    provider1 = AvatarProvider()
    provider2 = AvatarProvider()
    assert provider1 is provider2


def test_initialization_failure():
    with patch("providers.avatar_provider.open_zenoh_session") as mock_session:
        mock_session.side_effect = Exception("Connection failed")
        provider = AvatarProvider()

        assert not provider.running
        assert provider.session is None


def test_send_avatar_command(mock_zenoh):
    (
        mock_session,
        mock_session_instance,
        mock_publisher,
        mock_healthcheck_publisher,
        mock_subscriber,
    ) = mock_zenoh
    provider = AvatarProvider()

    result = provider.send_avatar_command("SMILE")

    assert result is True
    mock_publisher.put.assert_called_once()


def test_send_avatar_command_not_running(mock_zenoh):
    (
        mock_session,
        mock_session_instance,
        mock_publisher,
        mock_healthcheck_publisher,
        mock_subscriber,
    ) = mock_zenoh
    provider = AvatarProvider()
    provider.running = False

    result = provider.send_avatar_command("SMILE")

    assert result is False
    mock_publisher.put.assert_not_called()


def test_send_avatar_command_no_publisher(mock_zenoh):
    (
        mock_session,
        mock_session_instance,
        mock_publisher,
        mock_healthcheck_publisher,
        mock_subscriber,
    ) = mock_zenoh
    provider = AvatarProvider()
    provider.avatar_publisher = None

    result = provider.send_avatar_command("SMILE")

    assert result is False


def test_stop(mock_zenoh):
    (
        mock_session,
        mock_session_instance,
        mock_publisher,
        mock_healthcheck_publisher,
        mock_subscriber,
    ) = mock_zenoh
    provider = AvatarProvider()

    provider.stop()

    assert not provider.running
    mock_session_instance.close.assert_called_once()
