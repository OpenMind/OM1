import json
import pytest
from unittest.mock import MagicMock, patch

from src.providers.context_provider import ContextProvider


@pytest.fixture
def context_provider():
    with patch("src.providers.context_provider.open_zenoh_session") as mock_session:
        mock_publisher = MagicMock()
        mock_session.return_value.declare_publisher.return_value = mock_publisher

        provider = ContextProvider()
        provider.publisher = mock_publisher
        yield provider


def test_update_context_merges_state(context_provider):
    context_provider.update_context({"mode": "idle"})
    context_provider.update_context({"user": "test"})

    context = context_provider.get_context()

    assert context["mode"] == "idle"
    assert context["user"] == "test"


def test_set_context_field(context_provider):
    context_provider.set_context_field("volume", 5)

    context = context_provider.get_context()
    assert context["volume"] == 5


def test_context_publish_payload(context_provider):
    context_provider.update_context({"status": "active"})

    args, _ = context_provider.publisher.put.call_args
    payload = json.loads(args[0].decode("utf-8"))

    assert "context" in payload
    assert "timestamp" in payload
    assert payload["context"]["status"] == "active"


def test_clear_context(context_provider):
    context_provider.update_context({"foo": "bar"})
    context_provider.clear_context()

    context = context_provider.get_context()
    assert context == {}


def test_get_context_field_default(context_provider):
    value = context_provider.get_context_field("missing", default="x")
    assert value == "x"