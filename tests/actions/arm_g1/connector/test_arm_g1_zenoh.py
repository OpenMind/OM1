from unittest.mock import Mock, patch

import pytest

from actions.arm_g1.connector.zenoh import (
    CUSTOM_ACTION_MAP,
    SPORT_REQUEST_TOPIC,
    ARMZenohConnector,
)
from actions.arm_g1.interface import ArmAction, ArmInput
from actions.base import ActionConfig


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""
    with (
        patch("actions.arm_g1.connector.zenoh.open_zenoh_session") as mock_open_session,
        patch("actions.arm_g1.connector.zenoh.ZBytes") as mock_zbytes,
    ):
        mock_session = Mock()
        mock_open_session.return_value = mock_session
        mock_zbytes.side_effect = lambda x: x

        yield {
            "session": mock_session,
            "zbytes": mock_zbytes,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Create ARMZenohConnector with mocked dependencies."""
    config = ActionConfig()
    return ARMZenohConnector(config)


class TestARMZenohConnectorInit:
    """Test ARMZenohConnector initialization."""

    def test_init_opens_zenoh_session(self, connector, mock_dependencies):
        """Test that init opens a Zenoh session."""
        assert connector.session == mock_dependencies["session"]

    def test_init_handles_zenoh_error(self):
        """Test that init handles Zenoh session errors."""
        with (
            patch(
                "actions.arm_g1.connector.zenoh.open_zenoh_session"
            ) as mock_open_session,
            patch("actions.arm_g1.connector.zenoh.logging") as mock_logging,
        ):
            mock_open_session.side_effect = Exception("Connection refused")
            config = ActionConfig()
            conn = ARMZenohConnector(config)

            assert conn.session is None
            mock_logging.error.assert_called_once()
            assert "Connection refused" in str(mock_logging.error.call_args[0][0])


class TestARMZenohConnectorConnect:
    """Test connect method for custom arm actions."""

    @pytest.mark.asyncio
    async def test_connect_idle_returns_early(self, connector, mock_dependencies):
        """Test idle action returns without publishing."""
        arm_input = ArmInput(action=ArmAction.IDLE)
        await connector.connect(arm_input)
        mock_dependencies["session"].put.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_no_session(self):
        """Test connect with no Zenoh session logs error."""
        with (
            patch(
                "actions.arm_g1.connector.zenoh.open_zenoh_session"
            ) as mock_open_session,
            patch("actions.arm_g1.connector.zenoh.logging") as mock_logging,
        ):
            mock_open_session.side_effect = Exception("No connection")
            config = ActionConfig()
            conn = ARMZenohConnector(config)

            arm_input = ArmInput(action=ArmAction.SHAKE_HAND)
            await conn.connect(arm_input)

            mock_logging.error.assert_any_call(
                "ARMZenohConnector: No Zenoh session available"
            )

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector, mock_dependencies):
        """Test unknown action logs warning."""
        arm_input = ArmInput(action="unknown")  # type: ignore[arg-type]
        with patch("actions.arm_g1.connector.zenoh.logging") as mock_logging:
            await connector.connect(arm_input)
            mock_logging.warning.assert_called_once()
            assert "Unknown action" in str(mock_logging.warning.call_args[0][0])
            mock_dependencies["session"].put.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_shake_hand(self, connector, mock_dependencies):
        """Test shake hand publishes custom action."""
        arm_input = ArmInput(action=ArmAction.SHAKE_HAND)
        await connector.connect(arm_input)

        mock_dependencies["session"].put.assert_called_once()
        topic = mock_dependencies["session"].put.call_args[0][0]
        assert topic == SPORT_REQUEST_TOPIC

    @pytest.mark.asyncio
    async def test_connect_face_wave(self, connector, mock_dependencies):
        """Test face wave publishes custom action."""
        arm_input = ArmInput(action=ArmAction.FACE_WAVE)
        await connector.connect(arm_input)

        mock_dependencies["session"].put.assert_called_once()
        topic = mock_dependencies["session"].put.call_args[0][0]
        assert topic == SPORT_REQUEST_TOPIC

    @pytest.mark.asyncio
    async def test_connect_hands_up(self, connector, mock_dependencies):
        """Test hands up publishes custom action."""
        arm_input = ArmInput(action=ArmAction.HANDS_UP)
        await connector.connect(arm_input)

        mock_dependencies["session"].put.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_stand_still(self, connector, mock_dependencies):
        """Test stand still publishes custom action."""
        arm_input = ArmInput(action=ArmAction.STAND_STILL)
        await connector.connect(arm_input)

        mock_dependencies["session"].put.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_show_hand(self, connector, mock_dependencies):
        """Test show hand publishes custom action."""
        arm_input = ArmInput(action=ArmAction.SHOW_HAND)
        await connector.connect(arm_input)

        mock_dependencies["session"].put.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_all_custom_actions(self, connector, mock_dependencies):
        """Test all actions in CUSTOM_ACTION_MAP are publishable."""
        for action_value, expected_name in CUSTOM_ACTION_MAP.items():
            mock_dependencies["session"].put.reset_mock()
            arm_input = ArmInput(action=action_value)  # type: ignore[arg-type]
            await connector.connect(arm_input)
            mock_dependencies["session"].put.assert_called_once()


class TestARMZenohConnectorStop:
    """Test stop method."""

    def test_stop_closes_session(self, connector, mock_dependencies):
        """Test stop closes the Zenoh session."""
        connector.stop()
        mock_dependencies["session"].close.assert_called_once()
        assert connector.session is None

    def test_stop_no_session(self):
        """Test stop with no session does nothing."""
        with patch(
            "actions.arm_g1.connector.zenoh.open_zenoh_session"
        ) as mock_open_session:
            mock_open_session.side_effect = Exception("No connection")
            config = ActionConfig()
            conn = ARMZenohConnector(config)
            conn.stop()  # Should not raise
