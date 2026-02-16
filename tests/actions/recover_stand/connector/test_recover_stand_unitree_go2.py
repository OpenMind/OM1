from unittest.mock import Mock, patch

import pytest

from actions.recover_stand.connector.unitree_go2 import (
    RecoverStandUnitreeConfig,
    RecoverStandUnitreeConnector,
)
from actions.recover_stand.interface import RecoverAction, RecoverInput


MODULE = "actions.recover_stand.connector.unitree_go2"


@pytest.fixture
def mock_sport_client():
    """Mock SportClient for connector initialization."""
    with patch(f"{MODULE}.SportClient") as mock_sport:
        mock_instance = Mock()
        mock_sport.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def connector(mock_sport_client):
    """Create RecoverStandUnitreeConnector with mocked SportClient."""
    config = RecoverStandUnitreeConfig()
    return RecoverStandUnitreeConnector(config)


class TestRecoverStandUnitreeConnectorInit:
    """Test RecoverStandUnitreeConnector initialization."""

    def test_init_creates_sport_client(self, connector, mock_sport_client):
        """Test that initialization creates and configures SportClient."""
        mock_sport_client.SetTimeout.assert_called_once_with(10.0)
        mock_sport_client.Init.assert_called_once()
        assert connector.sport_client is mock_sport_client

    def test_init_handles_sport_client_error(self):
        """Test initialization when SportClient raises exception."""
        with (
            patch(f"{MODULE}.SportClient", side_effect=Exception("No robot")),
            patch(f"{MODULE}.logging") as mock_logging,
        ):
            config = RecoverStandUnitreeConfig()
            connector = RecoverStandUnitreeConnector(config)

            assert connector.sport_client is None
            mock_logging.error.assert_called()


class TestRecoverStandUnitreeConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_calls_recovery_stand(self, connector, mock_sport_client):
        """Test that connect calls RecoveryStand on sport client."""
        action_input = RecoverInput(action=RecoverAction.RECOVER)
        await connector.connect(action_input)
        mock_sport_client.RecoveryStand.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_logs_action(self, connector, mock_sport_client):
        """Test that connect logs the recovery action."""
        action_input = RecoverInput(action=RecoverAction.RECOVER)
        with patch(f"{MODULE}.logging") as mock_logging:
            await connector.connect(action_input)
            mock_logging.info.assert_any_call("Executing RecoveryStand command")

    @pytest.mark.asyncio
    async def test_connect_no_sport_client(self, connector):
        """Test connect with no sport client logs error."""
        connector.sport_client = None
        action_input = RecoverInput(action=RecoverAction.RECOVER)
        with patch(f"{MODULE}.logging") as mock_logging:
            await connector.connect(action_input)
            mock_logging.error.assert_any_call(
                "Cannot execute RecoveryStand: sport client not initialized"
            )

    @pytest.mark.asyncio
    async def test_connect_recovery_stand_error(self, connector, mock_sport_client):
        """Test connect when RecoveryStand raises exception."""
        mock_sport_client.RecoveryStand.side_effect = Exception("Hardware fault")
        action_input = RecoverInput(action=RecoverAction.RECOVER)
        with patch(f"{MODULE}.logging") as mock_logging:
            await connector.connect(action_input)
            mock_logging.error.assert_called()
