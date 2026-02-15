from unittest.mock import Mock, patch

import pytest

from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
from actions.arm_g1.interface import ArmAction, ArmInput
from actions.base import ActionConfig


@pytest.fixture
def mock_arm_client():
    """Mock G1ArmActionClient."""
    with patch("actions.arm_g1.connector.unitree_sdk.G1ArmActionClient") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def connector(mock_arm_client):
    """Create ARMUnitreeSDKConnector with mocked arm client."""
    config = ActionConfig()
    connector = ARMUnitreeSDKConnector(config)
    return connector


class TestARMUnitreeSDKConnectorInit:
    """Test ARMUnitreeSDKConnector initialization."""

    def test_init_creates_arm_client(self, mock_arm_client):
        """Test that init creates and initializes G1ArmActionClient."""
        config = ActionConfig()
        ARMUnitreeSDKConnector(config)

        mock_arm_client.SetTimeout.assert_called_once_with(10.0)
        mock_arm_client.Init.assert_called_once()

    def test_init_handles_client_error(self):
        """Test that init handles G1ArmActionClient initialization errors."""
        with (
            patch(
                "actions.arm_g1.connector.unitree_sdk.G1ArmActionClient"
            ) as mock_class,
            patch("actions.arm_g1.connector.unitree_sdk.logging") as mock_logging,
        ):
            mock_class.side_effect = Exception("Hardware not found")
            config = ActionConfig()
            ARMUnitreeSDKConnector(config)

            mock_logging.error.assert_called_once()
            assert "Hardware not found" in str(mock_logging.error.call_args[0][0])


class TestARMUnitreeSDKConnectorConnect:
    """Test connect method for each arm action."""

    @pytest.mark.asyncio
    async def test_connect_idle_returns_early(self, connector):
        """Test idle action returns without executing."""
        arm_input = ArmInput(action=ArmAction.IDLE)
        with patch("actions.arm_g1.connector.unitree_sdk.logging") as mock_logging:
            await connector.connect(arm_input)
            mock_logging.info.assert_any_call("No action to perform, returning.")
            connector.client.ExecuteAction.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_left_kiss(self, connector):
        """Test left kiss executes action ID 12."""
        arm_input = ArmInput(action=ArmAction.LEFT_KISS)
        await connector.connect(arm_input)
        connector.client.ExecuteAction.assert_called_once_with(12)

    @pytest.mark.asyncio
    async def test_connect_right_kiss(self, connector):
        """Test right kiss executes action ID 13."""
        arm_input = ArmInput(action=ArmAction.RIGHT_KISS)
        await connector.connect(arm_input)
        connector.client.ExecuteAction.assert_called_once_with(13)

    @pytest.mark.asyncio
    async def test_connect_clap(self, connector):
        """Test clap executes action ID 17."""
        arm_input = ArmInput(action=ArmAction.CLAP)
        await connector.connect(arm_input)
        connector.client.ExecuteAction.assert_called_once_with(17)

    @pytest.mark.asyncio
    async def test_connect_high_five(self, connector):
        """Test high five executes action ID 18."""
        arm_input = ArmInput(action=ArmAction.HIGH_FIVE)
        await connector.connect(arm_input)
        connector.client.ExecuteAction.assert_called_once_with(18)

    @pytest.mark.asyncio
    async def test_connect_shake_hand(self, connector):
        """Test shake hand executes action ID 27."""
        arm_input = ArmInput(action=ArmAction.SHAKE_HAND)
        await connector.connect(arm_input)
        connector.client.ExecuteAction.assert_called_once_with(27)

    @pytest.mark.asyncio
    async def test_connect_heart(self, connector):
        """Test heart executes action ID 20."""
        arm_input = ArmInput(action=ArmAction.HEART)
        await connector.connect(arm_input)
        connector.client.ExecuteAction.assert_called_once_with(20)

    @pytest.mark.asyncio
    async def test_connect_high_wave(self, connector):
        """Test high wave executes action ID 26."""
        arm_input = ArmInput(action=ArmAction.HIGH_WAVE)
        await connector.connect(arm_input)
        connector.client.ExecuteAction.assert_called_once_with(26)

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector):
        """Test unknown action logs warning and returns."""
        arm_input = ArmInput(action="unknown")  # type: ignore[arg-type]
        with patch("actions.arm_g1.connector.unitree_sdk.logging") as mock_logging:
            await connector.connect(arm_input)
            mock_logging.warning.assert_called_with("Unknown action: unknown")
            connector.client.ExecuteAction.assert_not_called()
