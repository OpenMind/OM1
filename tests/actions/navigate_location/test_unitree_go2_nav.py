# tests/actions/navigate_location/test_unitree_go2_nav_connector.py
"""Unit tests for the Unitree Go2 Navigation connector."""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from actions.navigate_location.interface import NavigateLocationInput

# Mock providers and zenoh_msgs before importing the connector
sys.modules["providers.unitree_go2_locations_provider"] = MagicMock()
sys.modules["providers.unitree_go2_navigation_provider"] = MagicMock()
sys.modules["providers.io_provider"] = MagicMock()
sys.modules["zenoh_msgs"] = MagicMock()


class TestUnitreeGo2NavConfig:
    """Tests for UnitreeGo2NavConfig."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from actions.navigate_location.connector.unitree_go2_nav import (
            UnitreeGo2NavConfig,
        )

        config = UnitreeGo2NavConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/list"
        assert config.timeout == 5
        assert config.refresh_interval == 30


class TestUnitreeGo2NavConnector:
    """Tests for UnitreeGo2NavConnector."""

    @pytest.fixture
    def mock_providers(self):
        """Set up mock providers for the connector."""
        mock_loc_provider = MagicMock()
        mock_nav_provider = MagicMock()
        mock_io_provider = MagicMock()

        return {
            "location_provider": mock_loc_provider,
            "navigation_provider": mock_nav_provider,
            "io_provider": mock_io_provider,
        }

    def test_connector_initialization(self, mock_providers):
        """Test connector initialization with providers."""
        with patch(
            "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2LocationsProvider",
            return_value=mock_providers["location_provider"],
        ):
            with patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2NavigationProvider",
                return_value=mock_providers["navigation_provider"],
            ):
                with patch(
                    "actions.navigate_location.connector.unitree_go2_nav.IOProvider",
                    return_value=mock_providers["io_provider"],
                ):
                    from actions.navigate_location.connector.unitree_go2_nav import (
                        UnitreeGo2NavConfig,
                        UnitreeGo2NavConnector,
                    )

                    config = UnitreeGo2NavConfig()
                    connector = UnitreeGo2NavConnector(config)

                    assert (
                        connector.location_provider
                        == mock_providers["location_provider"]
                    )
                    assert (
                        connector.navigation_provider
                        == mock_providers["navigation_provider"]
                    )
                    assert connector.io_provider == mock_providers["io_provider"]

    @pytest.mark.asyncio
    async def test_connect_location_found(self, mock_providers):
        """Test connect method when location is found."""
        mock_loc_provider = mock_providers["location_provider"]
        mock_nav_provider = mock_providers["navigation_provider"]

        # Mock finding a location
        location_data = {
            "name": "kitchen",
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
        mock_loc_provider.get_location.return_value = location_data

        with patch(
            "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2LocationsProvider",
            return_value=mock_loc_provider,
        ):
            with patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2NavigationProvider",
                return_value=mock_nav_provider,
            ):
                with patch(
                    "actions.navigate_location.connector.unitree_go2_nav.IOProvider",
                    return_value=mock_providers["io_provider"],
                ):
                    from actions.navigate_location.connector.unitree_go2_nav import (
                        UnitreeGo2NavConfig,
                        UnitreeGo2NavConnector,
                    )

                    config = UnitreeGo2NavConfig()
                    connector = UnitreeGo2NavConnector(config)

                    nav_input = NavigateLocationInput(action="go to the kitchen")
                    await connector.connect(nav_input)

                    # Check label cleaning
                    mock_loc_provider.get_location.assert_called_with("kitchen")
                    # Check navigation call
                    mock_nav_provider.publish_goal_pose.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_location_not_found(self, mock_providers, caplog):
        """Test connect method when location is not found."""
        mock_loc_provider = mock_providers["location_provider"]
        mock_loc_provider.get_location.return_value = None
        mock_loc_provider.get_all_locations.return_value = {
            "living room": {"name": "living room"}
        }

        with patch(
            "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2LocationsProvider",
            return_value=mock_loc_provider,
        ):
            with patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2NavigationProvider",
                return_value=mock_providers["navigation_provider"],
            ):
                with patch(
                    "actions.navigate_location.connector.unitree_go2_nav.IOProvider",
                    return_value=mock_providers["io_provider"],
                ):
                    from actions.navigate_location.connector.unitree_go2_nav import (
                        UnitreeGo2NavConfig,
                        UnitreeGo2NavConnector,
                    )

                    config = UnitreeGo2NavConfig()
                    connector = UnitreeGo2NavConnector(config)

                    nav_input = NavigateLocationInput(action="go to unknown")
                    with caplog.at_level(logging.WARNING):
                        await connector.connect(nav_input)

                    assert "Location 'unknown' not found" in caplog.text
                    mock_providers[
                        "navigation_provider"
                    ].publish_goal_pose.assert_not_called()

    @pytest.mark.parametrize(
        "prefix",
        [
            "go to the ",
            "go to ",
            "navigate to the ",
            "navigate to ",
            "move to the ",
            "move to ",
            "take me to the ",
            "take me to ",
        ],
    )
    @pytest.mark.asyncio
    async def test_label_cleaning(self, prefix, mock_providers):
        """Test that various prefixes are correctly removed from the label."""
        mock_loc_provider = mock_providers["location_provider"]
        mock_loc_provider.get_location.return_value = None

        with patch(
            "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2LocationsProvider",
            return_value=mock_loc_provider,
        ):
            with patch(
                "actions.navigate_location.connector.unitree_go2_nav.UnitreeGo2NavigationProvider",
                return_value=mock_providers["navigation_provider"],
            ):
                with patch(
                    "actions.navigate_location.connector.unitree_go2_nav.IOProvider",
                    return_value=mock_providers["io_provider"],
                ):
                    from actions.navigate_location.connector.unitree_go2_nav import (
                        UnitreeGo2NavConfig,
                        UnitreeGo2NavConnector,
                    )

                    config = UnitreeGo2NavConfig()
                    connector = UnitreeGo2NavConnector(config)

                    nav_input = NavigateLocationInput(action=f"{prefix}kitchen")
                    await connector.connect(nav_input)

                    mock_loc_provider.get_location.assert_called_with("kitchen")
