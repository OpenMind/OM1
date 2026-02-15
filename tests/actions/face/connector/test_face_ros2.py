from unittest.mock import patch

import pytest

from actions.base import ActionConfig
from actions.face.connector.ros2 import FaceRos2Connector
from actions.face.interface import FaceAction, FaceInput


@pytest.fixture
def connector():
    """Create a FaceRos2Connector with default config."""
    return FaceRos2Connector(ActionConfig())


class TestFaceRos2Connector:
    """Test the Face ROS2 connector."""

    def test_init(self):
        """Test initialization of FaceRos2Connector."""
        config = ActionConfig()
        connector = FaceRos2Connector(config)
        assert connector.config == config

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "face_action,expected_face",
        [
            (FaceAction.HAPPY, "happy"),
            (FaceAction.CONFUSED, "confused"),
            (FaceAction.CURIOUS, "curious"),
            (FaceAction.EXCITED, "excited"),
            (FaceAction.SAD, "sad"),
            (FaceAction.THINK, "think"),
        ],
    )
    async def test_connect_known_actions(self, connector, face_action, expected_face):
        """Test connect sends correct face dict to ROS2 for known actions."""
        face_input = FaceInput(action=face_action)
        with patch("actions.face.connector.ros2.logging") as mock_logging:
            await connector.connect(face_input)
            mock_logging.info.assert_called_with(
                f"SendThisToROS2: {{'face': '{expected_face}'}}"
            )

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector):
        """Test connect logs unknown face type for unrecognized action."""
        face_input = FaceInput(action=FaceAction.HAPPY)
        object.__setattr__(face_input, "action", "unknown_face")
        with patch("actions.face.connector.ros2.logging") as mock_logging:
            await connector.connect(face_input)
            calls = [str(c) for c in mock_logging.info.call_args_list]
            assert any("Unknown face type: unknown_face" in c for c in calls)
            assert any("SendThisToROS2: {'face': ''}" in c for c in calls)
