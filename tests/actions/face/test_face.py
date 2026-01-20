# tests/actions/face/test_connector.py
"""Unit tests for the Face action connector."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from actions.face.interface import FaceAction, FaceInput

# Mock the providers before importing the connector
sys.modules["providers.avatar_provider"] = MagicMock()


class TestFaceAvatarConnector:
    """Tests for FaceAvatarConnector."""

    @pytest.fixture
    def mock_avatar_provider(self):
        """Create a mock AvatarProvider."""
        mock_provider = MagicMock()
        mock_provider.send_avatar_command = MagicMock()
        mock_provider.stop = MagicMock()
        return mock_provider

    def test_connector_initialization(self, mock_avatar_provider):
        """Test connector initialization creates AvatarProvider."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            assert connector.avatar_provider is mock_avatar_provider

    @pytest.mark.asyncio
    async def test_connect_happy_face(self, mock_avatar_provider):
        """Test connect with HAPPY action sends Happy command."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            face_input = FaceInput(action=FaceAction.HAPPY)
            await connector.connect(face_input)

            mock_avatar_provider.send_avatar_command.assert_called_with("Happy")

    @pytest.mark.asyncio
    async def test_connect_sad_face(self, mock_avatar_provider):
        """Test connect with SAD action sends Sad command."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            face_input = FaceInput(action=FaceAction.SAD)
            await connector.connect(face_input)

            mock_avatar_provider.send_avatar_command.assert_called_with("Sad")

    @pytest.mark.asyncio
    async def test_connect_curious_face(self, mock_avatar_provider):
        """Test connect with CURIOUS action sends Curious command."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            face_input = FaceInput(action=FaceAction.CURIOUS)
            await connector.connect(face_input)

            mock_avatar_provider.send_avatar_command.assert_called_with("Curious")

    @pytest.mark.asyncio
    async def test_connect_confused_face(self, mock_avatar_provider):
        """Test connect with CONFUSED action sends Confused command."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            face_input = FaceInput(action=FaceAction.CONFUSED)
            await connector.connect(face_input)

            mock_avatar_provider.send_avatar_command.assert_called_with("Confused")

    @pytest.mark.asyncio
    async def test_connect_think_face(self, mock_avatar_provider):
        """Test connect with THINK action sends Think command."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            face_input = FaceInput(action=FaceAction.THINK)
            await connector.connect(face_input)

            mock_avatar_provider.send_avatar_command.assert_called_with("Think")

    @pytest.mark.asyncio
    async def test_connect_excited_face(self, mock_avatar_provider):
        """Test connect with EXCITED action sends Excited command."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            face_input = FaceInput(action=FaceAction.EXCITED)
            await connector.connect(face_input)

            mock_avatar_provider.send_avatar_command.assert_called_with("Excited")

    @pytest.mark.asyncio
    async def test_connect_unknown_action_logs_warning(
        self, mock_avatar_provider, caplog
    ):
        """Test connect with unknown action logs warning."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            # Create mock input with unknown action
            mock_input = MagicMock()
            mock_input.action = "unknown_action"

            with caplog.at_level(logging.WARNING):
                await connector.connect(mock_input)

            assert "Failed to send avatar face command" in caplog.text

    def test_stop_method(self, mock_avatar_provider):
        """Test stop method calls AvatarProvider.stop()."""
        with patch(
            "actions.face.connector.avatar.AvatarProvider",
            return_value=mock_avatar_provider,
        ):
            from actions.base import ActionConfig
            from actions.face.connector.avatar import FaceAvatarConnector

            config = ActionConfig()
            connector = FaceAvatarConnector(config)

            connector.stop()

            mock_avatar_provider.stop.assert_called_once()

    def test_connector_inherits_from_action_connector(self):
        """Test that FaceAvatarConnector inherits from ActionConnector."""
        from actions.base import ActionConnector
        from actions.face.connector.avatar import FaceAvatarConnector

        assert issubclass(FaceAvatarConnector, ActionConnector)
