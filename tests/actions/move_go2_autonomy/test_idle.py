from unittest.mock import patch

import pytest

from actions.base import ActionConfig
from actions.move_go2_autonomy.connector.idle import IDLEConnector
from actions.move_go2_autonomy.interface import MoveInput, MovementAction


@pytest.fixture
def connector():
    """Create an IDLEConnector with default config."""
    return IDLEConnector(ActionConfig())


class TestIDLEConnector:
    """Test the IDLE connector for Go2."""

    def test_init(self):
        """Test initialization of IDLEConnector."""
        config = ActionConfig()
        connector = IDLEConnector(config)
        assert connector.config == config

    @pytest.mark.asyncio
    async def test_connect_logs_and_returns(self, connector):
        """Test connect logs idle message and returns None."""
        move_input = MoveInput(action=MovementAction.STAND_STILL)
        with patch("actions.move_go2_autonomy.connector.idle.logging") as mock_logging:
            result = await connector.connect(move_input)
            mock_logging.info.assert_called_once_with(
                "IDLE connector called, doing nothing."
            )
            assert result is None
