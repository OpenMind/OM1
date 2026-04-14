from unittest.mock import AsyncMock, Mock, patch

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
            patch("actions.arm_g1.connector.zenoh.open_zenoh_session") as mock_open_session,
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
            patch("actions.arm_g1.connector.zenoh.open_zenoh_session") as mock_open_session,
            patch("actions.arm_g1.connector.zenoh.logging") as mock_logging,
        ):
            mock_open_session.side_effect = Exception("No connection")
            config = ActionConfig()
            conn = ARMZenohConnector(config)

            arm_input = ArmInput(action=ArmAction.SHAKE_HAND)
            await conn.connect(arm_input)

            mock_logging.error.assert_any_call("ARMZenohConnector: No Zenoh session available")

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
        with patch("actions.arm_g1.connector.zenoh.open_zenoh_session") as mock_open_session:
            mock_open_session.side_effect = Exception("No connection")
            config = ActionConfig()
            conn = ARMZenohConnector(config)
            conn.stop()  # Should not raise


class TestARMZenohConnectorAutoPayment:
    """Test automatic down_payment functionality."""

    @pytest.mark.asyncio
    async def test_do_payment_triggers_auto_down_payment(self, connector, mock_dependencies):
        """Test that do_payment action triggers automatic down_payment after 10 seconds."""
        arm_input = ArmInput(action=ArmAction.DO_PAYMENT)

        with patch.object(connector, "_auto_down_payment", new_callable=AsyncMock) as mock_auto_done:
            await connector.connect(arm_input)
            assert mock_dependencies["session"].put.call_count == 1
            mock_auto_done.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_down_payment_publishes_after_10_seconds(self, connector, mock_dependencies):
        """Test that _auto_down_payment publishes down_payment after 10 seconds."""
        # Mock asyncio.sleep to avoid actual delay in tests
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await connector._auto_down_payment()

            mock_sleep.assert_called_once_with(10)
            mock_dependencies["session"].put.assert_called_once()
            topic = mock_dependencies["session"].put.call_args[0][0]
            assert topic == SPORT_REQUEST_TOPIC

    @pytest.mark.asyncio
    async def test_auto_down_payment_no_session(self):
        """Test _auto_down_payment handles no session gracefully."""
        with (
            patch("actions.arm_g1.connector.zenoh.open_zenoh_session") as mock_open_session,
            patch("actions.arm_g1.connector.zenoh.logging") as mock_logging,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_open_session.side_effect = Exception("No connection")
            config = ActionConfig()
            conn = ARMZenohConnector(config)

            await conn._auto_down_payment()

            mock_logging.error.assert_any_call("ARMZenohConnector: No Zenoh session available for auto down_payment")

    @pytest.mark.asyncio
    async def test_auto_down_payment_exception_handling(self, connector):
        """Test that exceptions in _auto_down_payment are caught and logged."""
        with (
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("actions.arm_g1.connector.zenoh.logging") as mock_logging,
        ):
            mock_sleep.side_effect = Exception("Unexpected error")

            await connector._auto_down_payment()
            mock_logging.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_do_payment_full_workflow(self, connector, mock_dependencies):
        """Test complete workflow: do_payment publishes, then down_payment auto-publishes."""
        arm_input = ArmInput(action=ArmAction.DO_PAYMENT)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await connector.connect(arm_input)
            assert mock_dependencies["session"].put.call_count == 1
            await connector._auto_down_payment()
            assert mock_dependencies["session"].put.call_count == 2

    @pytest.mark.asyncio
    async def test_other_actions_dont_trigger_auto_payment(self, connector, mock_dependencies):
        """Test that non-payment actions don't trigger automatic down_payment."""
        test_actions = [
            ArmAction.SHAKE_HAND,
            ArmAction.FACE_WAVE,
            ArmAction.HANDS_UP,
            ArmAction.STAND_STILL,
        ]

        for action in test_actions:
            mock_dependencies["session"].put.reset_mock()
            arm_input = ArmInput(action=action)

            with patch.object(connector, "_auto_down_payment", new_callable=AsyncMock) as mock_auto_done:
                await connector.connect(arm_input)

                mock_dependencies["session"].put.assert_called_once()
                mock_auto_done.assert_not_called()

    @pytest.mark.asyncio
    async def test_down_payment_action_format(self, connector, mock_dependencies):
        """Test that down_payment action is formatted correctly."""
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await connector._auto_down_payment()

            call_args = mock_dependencies["session"].put.call_args
            topic = call_args[0][0]

            assert topic == SPORT_REQUEST_TOPIC
            assert mock_dependencies["session"].put.called
