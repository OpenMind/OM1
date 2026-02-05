"""Tests for the Face Avatar connector."""

import sys
from unittest.mock import MagicMock, Mock

# Mock modules at module load time BEFORE any other imports
sys.modules["zenoh"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()

from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from actions.base import ActionConfig  # noqa: E402
from actions.face.connector.avatar import FaceAvatarConnector  # noqa: E402
from actions.face.interface import FaceAction, FaceInput  # noqa: E402


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return ActionConfig()


@pytest.fixture
def face_input_happy():
    """Create a FaceInput instance with happy action."""
    return FaceInput(action=FaceAction.HAPPY)


@pytest.fixture
def face_input_sad():
    """Create a FaceInput instance with sad action."""
    return FaceInput(action=FaceAction.SAD)


@pytest.fixture
def face_input_curious():
    """Create a FaceInput instance with curious action."""
    return FaceInput(action=FaceAction.CURIOUS)


@pytest.fixture
def face_input_confused():
    """Create a FaceInput instance with confused action."""
    return FaceInput(action=FaceAction.CONFUSED)


@pytest.fixture
def face_input_think():
    """Create a FaceInput instance with think action."""
    return FaceInput(action=FaceAction.THINK)


@pytest.fixture
def face_input_excited():
    """Create a FaceInput instance with excited action."""
    return FaceInput(action=FaceAction.EXCITED)


class TestFaceAvatarConnector:
    """Test the Face Avatar connector."""

    @patch("actions.face.connector.avatar.AvatarProvider")
    def test_init(self, mock_avatar_provider_class, default_config):
        """Test initialization of FaceAvatarConnector."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)

        mock_avatar_provider_class.assert_called_once()
        assert connector.avatar_provider is not None
        assert connector.avatar_provider == mock_provider_instance

    @pytest.mark.asyncio
    @patch("actions.face.connector.avatar.AvatarProvider")
    async def test_connect_happy(
        self, mock_avatar_provider_class, default_config, face_input_happy
    ):
        """Test connect with happy expression."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)
        await connector.connect(face_input_happy)

        mock_provider_instance.send_avatar_command.assert_called_once_with("Happy")

    @pytest.mark.asyncio
    @patch("actions.face.connector.avatar.AvatarProvider")
    async def test_connect_sad(
        self, mock_avatar_provider_class, default_config, face_input_sad
    ):
        """Test connect with sad expression."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)
        await connector.connect(face_input_sad)

        mock_provider_instance.send_avatar_command.assert_called_once_with("Sad")

    @pytest.mark.asyncio
    @patch("actions.face.connector.avatar.AvatarProvider")
    async def test_connect_curious(
        self, mock_avatar_provider_class, default_config, face_input_curious
    ):
        """Test connect with curious expression."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)
        await connector.connect(face_input_curious)

        mock_provider_instance.send_avatar_command.assert_called_once_with("Curious")

    @pytest.mark.asyncio
    @patch("actions.face.connector.avatar.AvatarProvider")
    async def test_connect_confused(
        self, mock_avatar_provider_class, default_config, face_input_confused
    ):
        """Test connect with confused expression."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)
        await connector.connect(face_input_confused)

        mock_provider_instance.send_avatar_command.assert_called_once_with("Confused")

    @pytest.mark.asyncio
    @patch("actions.face.connector.avatar.AvatarProvider")
    async def test_connect_think(
        self, mock_avatar_provider_class, default_config, face_input_think
    ):
        """Test connect with think expression."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)
        await connector.connect(face_input_think)

        mock_provider_instance.send_avatar_command.assert_called_once_with("Think")

    @pytest.mark.asyncio
    @patch("actions.face.connector.avatar.AvatarProvider")
    async def test_connect_excited(
        self, mock_avatar_provider_class, default_config, face_input_excited
    ):
        """Test connect with excited expression."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)
        await connector.connect(face_input_excited)

        mock_provider_instance.send_avatar_command.assert_called_once_with("Excited")

    @patch("actions.face.connector.avatar.AvatarProvider")
    def test_stop(self, mock_avatar_provider_class, default_config):
        """Test stop method."""
        mock_provider_instance = Mock()
        mock_avatar_provider_class.return_value = mock_provider_instance

        connector = FaceAvatarConnector(default_config)
        connector.stop()

        mock_provider_instance.stop.assert_called_once()
