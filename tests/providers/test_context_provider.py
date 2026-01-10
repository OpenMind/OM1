"""
Unit tests for the ContextProvider in providers/context_provider.py.

Tests cover:
- Initialization behavior
- Context update functionality
- Single field updates
- Error handling when Zenoh is not available
- Stop/cleanup functionality
"""

import json
from unittest.mock import MagicMock, patch

from providers.context_provider import ContextProvider


class TestContextProviderInitialization:
    """Tests for ContextProvider initialization."""

    def test_initialization_sets_topic(self):
        """Test that initialization sets the correct context update topic."""
        with patch("providers.context_provider.open_zenoh_session"):
            provider = ContextProvider()
            assert provider.context_update_topic == "om/mode/context"
            ContextProvider.reset()

    def test_initialization_creates_session(self):
        """Test that initialization creates a Zenoh session."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            provider = ContextProvider()
            assert provider.session is mock_session
            assert provider.publisher is mock_publisher
            ContextProvider.reset()

    def test_initialization_handles_zenoh_error(self):
        """Test that initialization handles Zenoh errors gracefully."""
        with patch(
            "providers.context_provider.open_zenoh_session",
            side_effect=Exception("Zenoh connection failed"),
        ):
            provider = ContextProvider()
            assert provider.session is None
            assert provider.publisher is None
            ContextProvider.reset()


class TestContextProviderUpdateContext:
    """Tests for the update_context method."""

    def test_update_context_publishes_json(self):
        """Test that update_context publishes JSON-encoded context."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            provider = ContextProvider()
            context = {"key": "value", "count": 42}
            provider.update_context(context)

            mock_publisher.put.assert_called_once()
            call_args = mock_publisher.put.call_args[0][0]
            decoded = json.loads(call_args.decode("utf-8"))
            assert decoded == context
            ContextProvider.reset()

    def test_update_context_without_publisher(self):
        """Test that update_context handles missing publisher gracefully."""
        with patch(
            "providers.context_provider.open_zenoh_session",
            side_effect=Exception("No Zenoh"),
        ):
            provider = ContextProvider()
            # Should not raise, just log warning
            provider.update_context({"test": "data"})
            ContextProvider.reset()

    def test_update_context_handles_publish_error(self):
        """Test that update_context handles publish errors gracefully."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_publisher.put.side_effect = Exception("Publish failed")
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            provider = ContextProvider()
            # Should not raise, just log error
            provider.update_context({"test": "data"})
            ContextProvider.reset()


class TestContextProviderSetContextField:
    """Tests for the set_context_field method."""

    def test_set_context_field_calls_update_context(self):
        """Test that set_context_field delegates to update_context."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            provider = ContextProvider()
            provider.set_context_field("status", "active")

            mock_publisher.put.assert_called_once()
            call_args = mock_publisher.put.call_args[0][0]
            decoded = json.loads(call_args.decode("utf-8"))
            assert decoded == {"status": "active"}
            ContextProvider.reset()

    def test_set_context_field_with_various_types(self):
        """Test that set_context_field works with various value types."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            provider = ContextProvider()

            # Test with different types
            provider.set_context_field("string_val", "hello")
            provider.set_context_field("int_val", 123)
            provider.set_context_field("bool_val", True)
            provider.set_context_field("list_val", [1, 2, 3])

            assert mock_publisher.put.call_count == 4
            ContextProvider.reset()


class TestContextProviderStop:
    """Tests for the stop method."""

    def test_stop_closes_session(self):
        """Test that stop closes the Zenoh session."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            provider = ContextProvider()
            provider.stop()

            mock_session.close.assert_called_once()
            assert provider.session is None
            assert provider.publisher is None
            ContextProvider.reset()

    def test_stop_handles_close_error(self):
        """Test that stop handles session close errors gracefully."""
        mock_session = MagicMock()
        mock_session.close.side_effect = Exception("Close failed")
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        with patch(
            "providers.context_provider.open_zenoh_session", return_value=mock_session
        ):
            provider = ContextProvider()
            # Should not raise, just log error
            provider.stop()
            # Session should still be cleared
            assert provider.session is None
            assert provider.publisher is None
            ContextProvider.reset()

    def test_stop_without_session(self):
        """Test that stop handles missing session gracefully."""
        with patch(
            "providers.context_provider.open_zenoh_session",
            side_effect=Exception("No Zenoh"),
        ):
            provider = ContextProvider()
            # Should not raise
            provider.stop()
            ContextProvider.reset()
