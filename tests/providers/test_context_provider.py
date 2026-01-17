"""Tests for ContextProvider."""

import json
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock ALL external dependencies before any imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["elevenlabs"] = MagicMock()
sys.modules["riva"] = MagicMock()
sys.modules["riva.client"] = MagicMock()
sys.modules["pyaudio"] = MagicMock()
sys.modules["sounddevice"] = MagicMock()
sys.modules["rclpy"] = MagicMock()
sys.modules["sensor_msgs"] = MagicMock()
sys.modules["geometry_msgs"] = MagicMock()
sys.modules["nav_msgs"] = MagicMock()
sys.modules["std_msgs"] = MagicMock()


class TestContextProvider:
    """Tests for the ContextProvider class."""

    @pytest.fixture(autouse=True)
    def reset_modules(self):
        """Clear cached modules before each test."""
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith("providers")]
        for mod in modules_to_clear:
            del sys.modules[mod]
        yield
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith("providers")]
        for mod in modules_to_clear:
            del sys.modules[mod]

    def test_initialization_success(self):
        """Test successful initialization of ContextProvider."""
        with patch.dict("sys.modules", {"zenoh_msgs": MagicMock()}):
            mock_session = MagicMock()
            mock_publisher = MagicMock()
            mock_session.declare_publisher.return_value = mock_publisher

            with patch("zenoh_msgs.open_zenoh_session", return_value=mock_session):
                from providers.context_provider import ContextProvider

                if hasattr(ContextProvider, "reset"):
                    ContextProvider.reset()

                provider = ContextProvider()

                assert provider.context_update_topic == "om/mode/context"
                assert provider.session is not None

    def test_initialization_failure(self, caplog):
        """Test ContextProvider handles initialization failure gracefully."""
        mock_open = MagicMock(side_effect=Exception("Connection failed"))

        with patch.dict(
            "sys.modules", {"zenoh_msgs": MagicMock(open_zenoh_session=mock_open)}
        ):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            with caplog.at_level(logging.ERROR):
                provider = ContextProvider()

            assert provider.session is None
            assert provider.publisher is None

    def test_update_context_success(self):
        """Test successful context update."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        mock_zenoh_msgs = MagicMock()
        mock_zenoh_msgs.open_zenoh_session.return_value = mock_session

        with patch.dict("sys.modules", {"zenoh_msgs": mock_zenoh_msgs}):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            provider = ContextProvider()

            test_context = {"user_mood": "happy", "location": "home"}
            provider.update_context(test_context)

            expected_json = json.dumps(test_context).encode("utf-8")
            mock_publisher.put.assert_called_once_with(expected_json)

    def test_set_context_field(self):
        """Test set_context_field convenience method."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        mock_zenoh_msgs = MagicMock()
        mock_zenoh_msgs.open_zenoh_session.return_value = mock_session

        with patch.dict("sys.modules", {"zenoh_msgs": mock_zenoh_msgs}):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            provider = ContextProvider()
            provider.set_context_field("temperature", 72)

            expected_json = json.dumps({"temperature": 72}).encode("utf-8")
            mock_publisher.put.assert_called_once_with(expected_json)

    def test_stop_success(self):
        """Test successful stop of ContextProvider."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        mock_zenoh_msgs = MagicMock()
        mock_zenoh_msgs.open_zenoh_session.return_value = mock_session

        with patch.dict("sys.modules", {"zenoh_msgs": mock_zenoh_msgs}):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            provider = ContextProvider()
            provider.stop()

            mock_session.close.assert_called_once()
            assert provider.session is None
            assert provider.publisher is None

    def test_singleton_behavior(self):
        """Test that ContextProvider is a singleton."""
        mock_session = MagicMock()
        mock_publisher = MagicMock()
        mock_session.declare_publisher.return_value = mock_publisher

        mock_zenoh_msgs = MagicMock()
        mock_zenoh_msgs.open_zenoh_session.return_value = mock_session

        with patch.dict("sys.modules", {"zenoh_msgs": mock_zenoh_msgs}):
            from providers.context_provider import ContextProvider

            if hasattr(ContextProvider, "reset"):
                ContextProvider.reset()

            provider1 = ContextProvider()
            provider2 = ContextProvider()

            assert provider1 is provider2
