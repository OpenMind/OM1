from unittest.mock import patch

import pytest

from actions.dock_charging.connector.unitree_go2_dock import (
    UnitreeGo2DockConfig,
    UnitreeGo2DockConnector,
)
from actions.dock_charging.interface import DockCharging, DockChargingInput


def test_dock_charging_input_defaults():
    """Test DockChargingInput default values."""
    inp = DockChargingInput(action="dock")
    assert inp.action == "dock"
    assert inp.dock_location_name == "charging_dock"


def test_dock_charging_input_custom_location():
    """Test DockChargingInput with custom dock location name."""
    inp = DockChargingInput(action="dock", dock_location_name="dock2")
    assert inp.dock_location_name == "dock2"


def test_dock_charging_interface():
    """Test DockCharging interface structure."""
    inp = DockChargingInput(action="dock")
    interface = DockCharging(input=inp, output=inp)
    assert interface.input == inp
    assert interface.output == inp


def test_connector_config_defaults():
    """Test UnitreeGo2DockConfig default values."""
    config = UnitreeGo2DockConfig()
    assert config.base_url == "http://localhost:5000/maps/locations/list"
    assert config.timeout == 5
    assert config.refresh_interval == 30
    assert config.default_dock_location == "charging_dock"


def test_connector_config_custom():
    """Test UnitreeGo2DockConfig with custom values."""
    config = UnitreeGo2DockConfig(
        base_url="http://192.168.1.1:5000/maps/locations/list",
        timeout=10,
        refresh_interval=60,
        default_dock_location="my_dock",
    )
    assert config.base_url == "http://192.168.1.1:5000/maps/locations/list"
    assert config.timeout == 10
    assert config.refresh_interval == 60
    assert config.default_dock_location == "my_dock"


@pytest.fixture
def mock_connector():
    """Create a UnitreeGo2DockConnector with all providers mocked."""
    with (
        patch(
            "actions.dock_charging.connector.unitree_go2_dock.UnitreeGo2LocationsProvider"
        ) as mock_loc,
        patch(
            "actions.dock_charging.connector.unitree_go2_dock.UnitreeGo2NavigationProvider"
        ) as mock_nav,
        patch(
            "actions.dock_charging.connector.unitree_go2_dock.UnitreeGo2ChargingProvider"
        ) as mock_charge,
    ):
        mock_charge.return_value.get_charging_status.return_value = 0
        config = UnitreeGo2DockConfig()
        connector = UnitreeGo2DockConnector(config=config)
        connector.location_provider = mock_loc.return_value
        connector.navigation_provider = mock_nav.return_value
        connector.charging_provider = mock_charge.return_value
        yield connector


@pytest.mark.asyncio
async def test_connect_already_docked(mock_connector):
    """Test connect skips navigation if robot is already docked."""
    mock_connector.charging_provider.get_charging_status.return_value = 1

    inp = DockChargingInput(action="dock")
    await mock_connector.connect(inp)

    mock_connector.navigation_provider.publish_goal_pose.assert_not_called()


@pytest.mark.asyncio
async def test_connect_location_not_found(mock_connector):
    """Test connect logs warning if dock location is not found."""
    mock_connector.charging_provider.get_charging_status.return_value = 0
    mock_connector.location_provider.get_location.return_value = None
    mock_connector.location_provider.get_all_locations.return_value = {}

    inp = DockChargingInput(action="dock")
    await mock_connector.connect(inp)

    mock_connector.navigation_provider.publish_goal_pose.assert_not_called()


@pytest.mark.asyncio
async def test_connect_navigates_to_dock(mock_connector):
    """Test connect publishes navigation goal when dock location is found."""
    mock_connector.charging_provider.get_charging_status.return_value = 0
    mock_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock")
    await mock_connector.connect(inp)

    mock_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_connect_uses_custom_dock_location(mock_connector):
    """Test connect uses dock_location_name from input over config default."""
    mock_connector.charging_provider.get_charging_status.return_value = 0
    mock_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 3.0, "y": 4.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock", dock_location_name="dock2")
    await mock_connector.connect(inp)

    mock_connector.location_provider.get_location.assert_called_with("dock2")
    mock_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_connect_uses_config_default_when_no_location_in_input(mock_connector):
    """Test connect falls back to config default_dock_location when input has None."""
    mock_connector.charging_provider.get_charging_status.return_value = 0
    mock_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 1.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock", dock_location_name=None)
    await mock_connector.connect(inp)

    mock_connector.location_provider.get_location.assert_called_with("charging_dock")


@pytest.mark.asyncio
async def test_connect_handles_navigation_exception(mock_connector):
    """Test connect handles exception from navigation provider gracefully."""
    mock_connector.charging_provider.get_charging_status.return_value = 0
    mock_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }
    mock_connector.navigation_provider.publish_goal_pose.side_effect = Exception(
        "Navigation error"
    )

    inp = DockChargingInput(action="dock")
    # Should not raise exception
    await mock_connector.connect(inp)
