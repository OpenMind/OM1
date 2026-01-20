import json
from unittest.mock import MagicMock, patch

import pytest

from providers.context_provider import ContextProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    ContextProvider.reset()  # type: ignore
    yield

    try:
        provider = ContextProvider()
        provider.stop()
    except Exception:
        pass

    ContextProvider.reset()  # type: ignore


@pytest.fixture
def mock_zenoh():
    with patch("providers.context_provider.open_zenoh_session") as mock_session:
        mock_session_instance = MagicMock()
        mock_publisher = MagicMock()
        mock_session_instance.declare_publisher.return_value = mock_publisher
        mock_session.return_value = mock_session_instance
        yield mock_session, mock_session_instance, mock_publisher


def test_initialization(mock_zenoh):
    _, mock_session_instance, mock_publisher = mock_zenoh
    provider = ContextProvider()

    assert provider.context_update_topic == "om/mode/context"
    assert provider.session == mock_session_instance
    assert provider.publisher == mock_publisher
    mock_session_instance.declare_publisher.assert_called_once_with("om/mode/context")


def test_singleton_pattern():
    provider1 = ContextProvider()
    provider2 = ContextProvider()
    assert provider1 is provider2


def test_update_context(mock_zenoh):
    _, _, mock_publisher = mock_zenoh
    provider = ContextProvider()

    context = {"location": "kitchen", "task": "cooking"}
    provider.update_context(context)

    expected_json = json.dumps(context)
    mock_publisher.put.assert_called_once_with(expected_json.encode("utf-8"))


def test_update_context_no_publisher(mock_zenoh):
    _, _, mock_publisher = mock_zenoh
    provider = ContextProvider()
    provider.publisher = None

    context = {"location": "kitchen"}
    provider.update_context(context)

    mock_publisher.put.assert_not_called()


def test_set_context_field(mock_zenoh):
    _, _, mock_publisher = mock_zenoh
    provider = ContextProvider()

    provider.set_context_field("battery_level", 75)

    expected_json = json.dumps({"battery_level": 75})
    mock_publisher.put.assert_called_once_with(expected_json.encode("utf-8"))


def test_stop(mock_zenoh):
    _, mock_session_instance, _ = mock_zenoh
    provider = ContextProvider()

    provider.stop()

    mock_session_instance.close.assert_called_once()


def test_initialization_failure():
    with patch("providers.context_provider.open_zenoh_session") as mock_session:
        mock_session.side_effect = Exception("Connection failed")
        provider = ContextProvider()

        assert provider.session is None
        assert provider.publisher is None
