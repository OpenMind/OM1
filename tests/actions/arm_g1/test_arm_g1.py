# tests/actions/arm_g1/test_arm_g1_connector.py
"""Unit tests for the G1 Arm action connector."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from actions.arm_g1.interface import ArmAction, ArmInput

# Mock unitree_sdk before importing the connector
sys.modules["unitree"] = MagicMock()
sys.modules["unitree.unitree_sdk2py"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.g1"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.g1.arm"] = MagicMock()
sys.modules["unitree.unitree_sdk2py.g1.arm.g1_arm_action_client"] = MagicMock()


class TestARMG1Connector:
    """Tests for ARMUnitreeSDKConnector."""

    @pytest.fixture
    def mock_arm_client(self):
        """Create a mock G1ArmActionClient."""
        mock_client = MagicMock()
        return mock_client

    def test_connector_initialization(self, mock_arm_client):
        """Test connector initialization and setup."""
        with patch(
            "actions.arm_g1.connector.unitree_sdk.G1ArmActionClient",
            return_value=mock_arm_client,
        ):
            from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
            from actions.base import ActionConfig

            config = ActionConfig()
            connector = ARMUnitreeSDKConnector(config)

            # Verify client setup methods called
            mock_arm_client.SetTimeout.assert_called_with(10.0)
            mock_arm_client.Init.assert_called_once()
            assert connector.client == mock_arm_client

    def test_connector_init_failure_logs_error(self, caplog):
        """Test that initialization error is caught and logged."""
        with patch(
            "actions.arm_g1.connector.unitree_sdk.G1ArmActionClient",
            side_effect=Exception("Connection failed"),
        ):
            from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
            from actions.base import ActionConfig

            with caplog.at_level(logging.ERROR):
                config = ActionConfig()
                _ = ARMUnitreeSDKConnector(config)

            assert (
                "Failed to initialize G1 Arm Action Client: Connection failed"
                in caplog.text
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "action,expected_id",
        [
            (ArmAction.LEFT_KISS, 12),
            (ArmAction.RIGHT_KISS, 13),
            (ArmAction.CLAP, 17),
            (ArmAction.HIGH_FIVE, 18),
            (ArmAction.SHAKE_HAND, 27),
            (ArmAction.HEART, 20),
            (ArmAction.HIGH_WAVE, 26),
        ],
    )
    async def test_connect_valid_actions(self, mock_arm_client, action, expected_id):
        """Test that valid arm actions translate to correct IDs."""
        with patch(
            "actions.arm_g1.connector.unitree_sdk.G1ArmActionClient",
            return_value=mock_arm_client,
        ):
            from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
            from actions.base import ActionConfig

            config = ActionConfig()
            connector = ARMUnitreeSDKConnector(config)

            arm_input = ArmInput(action=action)
            await connector.connect(arm_input)

            mock_arm_client.ExecuteAction.assert_called_with(expected_id)

    @pytest.mark.asyncio
    async def test_connect_idle_action_does_nothing(self, mock_arm_client):
        """Test that IDLE action does not trigger an execution."""
        with patch(
            "actions.arm_g1.connector.unitree_sdk.G1ArmActionClient",
            return_value=mock_arm_client,
        ):
            from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
            from actions.base import ActionConfig

            config = ActionConfig()
            connector = ARMUnitreeSDKConnector(config)

            arm_input = ArmInput(action=ArmAction.IDLE)
            await connector.connect(arm_input)

            mock_arm_client.ExecuteAction.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_unknown_action_logs_warning(self, mock_arm_client, caplog):
        """Test that unknown action logs a warning and does nothing."""
        with patch(
            "actions.arm_g1.connector.unitree_sdk.G1ArmActionClient",
            return_value=mock_arm_client,
        ):
            from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
            from actions.base import ActionConfig

            config = ActionConfig()
            connector = ARMUnitreeSDKConnector(config)

            # Create an input with a non-existent action
            mock_input = MagicMock()
            mock_input.action = "moonwalk"

            with caplog.at_level(logging.WARNING):
                await connector.connect(mock_input)

            assert "Unknown action: moonwalk" in caplog.text
            mock_arm_client.ExecuteAction.assert_not_called()

    def test_connector_inherits_from_action_connector(self):
        """Test that ARMUnitreeSDKConnector inherits from ActionConnector."""
        from actions.arm_g1.connector.unitree_sdk import ARMUnitreeSDKConnector
        from actions.base import ActionConnector

        assert issubclass(ARMUnitreeSDKConnector, ActionConnector)
