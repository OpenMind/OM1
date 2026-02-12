from unittest.mock import Mock, patch

import pytest

from actions.move_go2_action.connector.unitree_sdk import (
    ActionUnitreeSDKConfig,
    ActionUnitreeSDKConnector,
)
from actions.move_go2_action.interface import Action, ActionInput


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for connector initialization."""

    with (
        patch(
            "actions.move_go2_action.connector.unitree_sdk.UnitreeGo2RPLidarProvider"
        ) as mock_lidar,
        patch(
            "actions.move_go2_action.connector.unitree_sdk.UnitreeGo2StateProvider"
        ) as mock_state,
        patch(
            "actions.move_go2_action.connector.unitree_sdk.SportClient"
        ) as mock_sport,
        patch(
            "actions.move_go2_action.connector.unitree_sdk.UnitreeGo2OdomProvider"
        ) as mock_odom,
    ):
        mock_lidar_instance = Mock()
        mock_lidar.return_value = mock_lidar_instance

        mock_state_instance = Mock()
        mock_state_instance.go2_action_progress = 0
        mock_state.return_value = mock_state_instance

        mock_sport_instance = Mock()
        mock_sport.return_value = mock_sport_instance

        mock_odom_instance = Mock()
        mock_odom.return_value = mock_odom_instance

        yield {
            "lidar": mock_lidar_instance,
            "state": mock_state_instance,
            "sport": mock_sport_instance,
            "odom": mock_odom_instance,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Create ActionUnitreeSDKConnector with mocked dependencies."""
    config = ActionUnitreeSDKConfig()
    return ActionUnitreeSDKConnector(config)


class TestActionUnitreeSDKConfig:
    """Test ActionUnitreeSDKConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ActionUnitreeSDKConfig()
        assert config.unitree_ethernet is None

    def test_custom_ethernet(self):
        """Test custom ethernet configuration."""
        config = ActionUnitreeSDKConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"


class TestActionUnitreeSDKConnectorInit:
    """Test ActionUnitreeSDKConnector initialization."""

    def test_initialization(self, connector, mock_dependencies):
        """Test successful initialization."""
        assert connector.turn_speed == 0.8
        assert connector.angle_tolerance == 5.0
        assert connector.distance_tolerance == 0.05
        assert connector.movement_attempts == 0
        assert connector.movement_attempt_limit == 15

        mock_dependencies["sport"].SetTimeout.assert_called_once_with(10.0)
        mock_dependencies["sport"].Init.assert_called_once()
        mock_dependencies["sport"].StopMove.assert_called_once()

    def test_initialization_sport_client_error(self):
        """Test initialization when SportClient fails."""
        with (
            patch(
                "actions.move_go2_action.connector.unitree_sdk.UnitreeGo2RPLidarProvider"
            ),
            patch(
                "actions.move_go2_action.connector.unitree_sdk.UnitreeGo2StateProvider"
            ),
            patch(
                "actions.move_go2_action.connector.unitree_sdk.SportClient"
            ) as mock_sport,
            patch(
                "actions.move_go2_action.connector.unitree_sdk.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.move_go2_action.connector.unitree_sdk.logging"
            ) as mock_logging,
        ):
            mock_sport.side_effect = Exception("Sport client init failed")
            config = ActionUnitreeSDKConfig()
            connector = ActionUnitreeSDKConnector(config)

            assert connector.sport_client is None
            mock_logging.error.assert_called()


class TestActionUnitreeSDKConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_stand_still(self, connector, mock_dependencies):
        """Test stand still action."""
        action_input = ActionInput(action=Action.STAND_STILL)
        with patch(
            "actions.move_go2_action.connector.unitree_sdk.logging"
        ) as mock_logging:
            await connector.connect(action_input)
            mock_logging.info.assert_any_call(
                "ActionUnitreeSDKConnector: Standing still"
            )

    @pytest.mark.asyncio
    async def test_connect_shake_paw_ready(self, connector, mock_dependencies):
        """Test shake paw when robot is ready (progress == 0)."""
        mock_dependencies["state"].go2_action_progress = 0
        action_input = ActionInput(action=Action.SHAKE_PAW)
        await connector.connect(action_input)
        mock_dependencies["sport"].Hello.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_shake_paw_busy(self, connector, mock_dependencies):
        """Test shake paw when robot is busy (progress != 0)."""
        mock_dependencies["state"].go2_action_progress = 50
        action_input = ActionInput(action=Action.SHAKE_PAW)
        with patch(
            "actions.move_go2_action.connector.unitree_sdk.logging"
        ) as mock_logging:
            await connector.connect(action_input)
            mock_logging.info.assert_any_call(
                "ActionUnitreeSDKConnector: Still performing previous action"
            )
            mock_dependencies["sport"].Hello.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_shake_paw_error(self, connector, mock_dependencies):
        """Test shake paw when sport client raises exception."""
        mock_dependencies["state"].go2_action_progress = 0
        mock_dependencies["sport"].Hello.side_effect = Exception("Hardware error")
        action_input = ActionInput(action=Action.SHAKE_PAW)
        with patch(
            "actions.move_go2_action.connector.unitree_sdk.logging"
        ) as mock_logging:
            await connector.connect(action_input)
            mock_logging.error.assert_called()

    @pytest.mark.asyncio
    async def test_connect_dance_ready(self, connector, mock_dependencies):
        """Test dance when robot is ready."""
        mock_dependencies["state"].go2_action_progress = 0
        action_input = ActionInput(action=Action.DANCE)
        with patch(
            "actions.move_go2_action.connector.unitree_sdk.random"
        ) as mock_random:
            mock_random.choice.return_value = mock_dependencies["sport"].Dance1
            await connector.connect(action_input)
            mock_dependencies["sport"].Dance1.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_stretch_ready(self, connector, mock_dependencies):
        """Test stretch when robot is ready."""
        mock_dependencies["state"].go2_action_progress = 0
        action_input = ActionInput(action=Action.STRETCH)
        await connector.connect(action_input)
        mock_dependencies["sport"].Stretch.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_stretch_busy(self, connector, mock_dependencies):
        """Test stretch when robot is busy."""
        mock_dependencies["state"].go2_action_progress = 75
        action_input = ActionInput(action=Action.STRETCH)
        with patch(
            "actions.move_go2_action.connector.unitree_sdk.logging"
        ) as mock_logging:
            await connector.connect(action_input)
            mock_logging.info.assert_any_call(
                "ActionUnitreeSDKConnector: Still performing previous action"
            )
            mock_dependencies["sport"].Stretch.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector, mock_dependencies):
        """Test unknown action logs warning."""
        action_input = ActionInput(action="fly")  # type: ignore[arg-type]
        with patch(
            "actions.move_go2_action.connector.unitree_sdk.logging"
        ) as mock_logging:
            await connector.connect(action_input)
            mock_logging.warning.assert_called_with(
                "Action 'fly' not recognized or not implemented."
            )

    @pytest.mark.asyncio
    async def test_connect_no_sport_client(self, connector, mock_dependencies):
        """Test shake paw with no sport client does not crash."""
        mock_dependencies["state"].go2_action_progress = 0
        connector.sport_client = None
        action_input = ActionInput(action=Action.SHAKE_PAW)
        await connector.connect(action_input)
