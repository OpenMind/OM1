import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.navigate_location.interface import NavigateLocationInput

sys.modules["om1_speech"] = MagicMock()

from actions.navigate_location.connector.unitree_g1_nav import (  # noqa: E402
    UnitreeG1NavConfig,
    UnitreeG1NavConnector,
)
from actions.navigate_location.connector.unitree_go2_nav import (  # noqa: E402
    UnitreeGo2NavConfig,
    UnitreeGo2NavConnector,
)


class TestUnitreeG1NavConfig:
    """Test UnitreeG1NavConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeG1NavConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/list"
        assert config.timeout == 5
        assert config.refresh_interval == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnitreeG1NavConfig(
            base_url="http://custom:8080/locations",
            timeout=10,
            refresh_interval=60,
        )
        assert config.base_url == "http://custom:8080/locations"
        assert config.timeout == 10
        assert config.refresh_interval == 60


class TestUnitreeG1NavConnectorInit:
    """Test UnitreeG1NavConnector initialization."""

    def test_init(self):
        """Test successful initialization."""
        with (
            patch(
                "actions.navigate_location.connector.unitree_g1_nav.UnitreeG1LocationsProvider"
            ) as mock_loc,
            patch(
                "actions.navigate_location.connector.unitree_g1_nav.UnitreeG1NavigationProvider"
            ) as mock_nav,
            patch(
                "actions.navigate_location.connector.unitree_g1_nav.IOProvider"
            ) as mock_io,
        ):
            config = UnitreeG1NavConfig()
            connector = UnitreeG1NavConnector(config)

            mock_loc.assert_called_once_with(
                "http://localhost:5000/maps/locations/list", 5, 30
            )
            mock_nav.assert_called_once()
            mock_io.assert_called_once()
            assert connector.location_provider == mock_loc.return_value
            assert connector.navigation_provider == mock_nav.return_value


class TestUnitreeG1NavConnectorConnect:
    """Test G1 connect method."""

    @pytest.fixture
    def g1_connector(self):
        """Create G1 nav connector with mocked dependencies."""
        with (
            patch(
                "actions.navigate_location.connector.unitree_g1_nav.UnitreeG1LocationsProvider"
            ) as mock_loc,
            patch(
                "actions.navigate_location.connector.unitree_g1_nav.UnitreeG1NavigationProvider"
            ) as mock_nav,
            patch("actions.navigate_location.connector.unitree_g1_nav.IOProvider"),
        ):
            mock_loc_instance = Mock()
            mock_loc.return_value = mock_loc_instance

            mock_nav_instance = Mock()
            mock_nav.return_value = mock_nav_instance

            config = UnitreeG1NavConfig()
            connector = UnitreeG1NavConnector(config)
            yield connector, mock_loc_instance, mock_nav_instance

    @pytest.mark.asyncio
    async def test_connect_location_not_found_no_locations(self, g1_connector):
        """Test connect when location is not found and no locations available."""
        connector, mock_loc, mock_nav = g1_connector
        mock_loc.get_location.return_value = None
        mock_loc.get_all_locations.return_value = {}

        nav_input = NavigateLocationInput(action="unknown_place")
        with patch(
            "actions.navigate_location.connector.unitree_g1_nav.logging"
        ) as mock_logging:
            await connector.connect(nav_input)
            mock_logging.warning.assert_called_with(
                "Location 'unknown_place' not found. No locations available."
            )
            mock_nav.publish_goal_pose.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_location_not_found_with_available(self, g1_connector):
        """Test connect when location is not found but others are available."""
        connector, mock_loc, mock_nav = g1_connector
        mock_loc.get_location.return_value = None
        mock_loc.get_all_locations.return_value = {
            "kitchen": {"name": "kitchen"},
            "office": {"name": "office"},
        }

        nav_input = NavigateLocationInput(action="bedroom")
        with patch(
            "actions.navigate_location.connector.unitree_g1_nav.logging"
        ) as mock_logging:
            await connector.connect(nav_input)
            warning_msg = mock_logging.warning.call_args[0][0]
            assert "bedroom" in warning_msg
            assert "kitchen" in warning_msg

    @pytest.mark.asyncio
    async def test_connect_location_found(self, g1_connector):
        """Test connect when location is found."""
        connector, mock_loc, mock_nav = g1_connector
        mock_loc.get_location.return_value = {
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.5, "w": 0.866},
            }
        }

        nav_input = NavigateLocationInput(action="kitchen")
        with patch(
            "actions.navigate_location.connector.unitree_g1_nav.logging"
        ) as mock_logging:
            await connector.connect(nav_input)
            mock_nav.publish_goal_pose.assert_called_once()
            mock_logging.info.assert_any_call("Navigation to 'kitchen' initiated")

    @pytest.mark.asyncio
    async def test_connect_strips_prefix(self, g1_connector):
        """Test connect strips 'go to' prefix from action."""
        connector, mock_loc, mock_nav = g1_connector
        mock_loc.get_location.return_value = None
        mock_loc.get_all_locations.return_value = {}

        nav_input = NavigateLocationInput(action="Go to the kitchen")
        with patch("actions.navigate_location.connector.unitree_g1_nav.logging"):
            await connector.connect(nav_input)
            mock_loc.get_location.assert_called_with("kitchen")

    @pytest.mark.asyncio
    async def test_connect_publish_error(self, g1_connector):
        """Test connect handles publish error."""
        connector, mock_loc, mock_nav = g1_connector
        mock_loc.get_location.return_value = {
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        }
        mock_nav.publish_goal_pose.side_effect = Exception("Publish failed")

        nav_input = NavigateLocationInput(action="kitchen")
        with patch(
            "actions.navigate_location.connector.unitree_g1_nav.logging"
        ) as mock_logging:
            await connector.connect(nav_input)
            mock_logging.error.assert_called()


class TestUnitreeGo2NavConfig:
    """Test UnitreeGo2NavConfig configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = UnitreeGo2NavConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/list"
        assert config.timeout == 5
        assert config.refresh_interval == 30

    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnitreeGo2NavConfig(
            base_url="http://go2:9090/locations",
            timeout=15,
            refresh_interval=45,
        )
        assert config.base_url == "http://go2:9090/locations"
        assert config.timeout == 15
        assert config.refresh_interval == 45


class TestUnitreeGo2NavConnectorInit:
    """Test UnitreeGo2NavConnector initialization."""

    def test_init(self):
        """Test successful initialization."""
        with (
            patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2LocationsProvider"
            ) as mock_loc,
            patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2NavigationProvider"
            ) as mock_nav,
            patch(
                "actions.navigate_location.connector.unitree_go2_nav.IOProvider"
            ) as mock_io,
        ):
            config = UnitreeGo2NavConfig()
            connector = UnitreeGo2NavConnector(config)

            mock_loc.assert_called_once_with(
                "http://localhost:5000/maps/locations/list", 5, 30
            )
            mock_nav.assert_called_once()
            mock_io.assert_called_once()
            assert connector.location_provider == mock_loc.return_value


class TestUnitreeGo2NavConnectorConnect:
    """Test Go2 connect method."""

    @pytest.fixture
    def go2_connector(self):
        """Create Go2 nav connector with mocked dependencies."""
        with (
            patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2LocationsProvider"
            ) as mock_loc,
            patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2NavigationProvider"
            ) as mock_nav,
            patch("actions.navigate_location.connector.unitree_go2_nav.IOProvider"),
        ):
            mock_loc_instance = Mock()
            mock_loc.return_value = mock_loc_instance

            mock_nav_instance = Mock()
            mock_nav.return_value = mock_nav_instance

            config = UnitreeGo2NavConfig()
            connector = UnitreeGo2NavConnector(config)
            yield connector, mock_loc_instance, mock_nav_instance

    @pytest.mark.asyncio
    async def test_connect_location_found(self, go2_connector):
        """Test connect when location is found."""
        connector, mock_loc, mock_nav = go2_connector
        mock_loc.get_location.return_value = {
            "pose": {
                "position": {"x": 3.0, "y": 4.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        }

        nav_input = NavigateLocationInput(action="sofa")
        with patch(
            "actions.navigate_location.connector.unitree_go2_nav.logging"
        ) as mock_logging:
            await connector.connect(nav_input)
            mock_nav.publish_goal_pose.assert_called_once()
            mock_logging.info.assert_any_call("Navigation to 'sofa' initiated")

    @pytest.mark.asyncio
    async def test_connect_location_not_found(self, go2_connector):
        """Test connect when location is not found."""
        connector, mock_loc, mock_nav = go2_connector
        mock_loc.get_location.return_value = None
        mock_loc.get_all_locations.return_value = {"table": {"name": "table"}}

        nav_input = NavigateLocationInput(action="garden")
        with patch(
            "actions.navigate_location.connector.unitree_go2_nav.logging"
        ) as mock_logging:
            await connector.connect(nav_input)
            warning_msg = mock_logging.warning.call_args[0][0]
            assert "garden" in warning_msg
            assert "table" in warning_msg

    @pytest.mark.asyncio
    async def test_connect_strips_navigate_prefix(self, go2_connector):
        """Test connect strips 'navigate to' prefix."""
        connector, mock_loc, mock_nav = go2_connector
        mock_loc.get_location.return_value = None
        mock_loc.get_all_locations.return_value = {}

        nav_input = NavigateLocationInput(action="Navigate to the bedroom")
        await connector.connect(nav_input)
        mock_loc.get_location.assert_called_with("bedroom")

    @pytest.mark.asyncio
    async def test_connect_location_with_empty_pose(self, go2_connector):
        """Test connect with location that has empty/no pose data."""
        connector, mock_loc, mock_nav = go2_connector
        mock_loc.get_location.return_value = {"pose": None}

        nav_input = NavigateLocationInput(action="hall")
        await connector.connect(nav_input)
        mock_nav.publish_goal_pose.assert_called_once()
