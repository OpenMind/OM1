from unittest.mock import patch

import pytest

from actions.dock_charging.connector.unitree_g1_dock import (
    UnitreeG1DockConfig,
    UnitreeG1DockConnector,
)
from actions.dock_charging.connector.unitree_go2_dock import (
    UnitreeGo2DockConfig,
    UnitreeGo2DockConnector,
)
from actions.dock_charging.interface import DockCharging, DockChargingInput

# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Go2 config tests
# ---------------------------------------------------------------------------


def test_go2_connector_config_defaults():
    """Test UnitreeGo2DockConfig default values."""
    config = UnitreeGo2DockConfig()
    assert config.base_url == "http://localhost:5000/maps/locations/list"
    assert config.timeout == 5
    assert config.refresh_interval == 30
    assert config.default_dock_location == "charging_dock"


def test_go2_connector_config_custom():
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


# ---------------------------------------------------------------------------
# Go2 connector tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_go2_connector():
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
async def test_go2_connect_already_docked(mock_go2_connector):
    """Test connect skips navigation if robot is already docked."""
    mock_go2_connector.charging_provider.get_charging_status.return_value = 1

    inp = DockChargingInput(action="dock")
    await mock_go2_connector.connect(inp)

    mock_go2_connector.navigation_provider.publish_goal_pose.assert_not_called()


@pytest.mark.asyncio
async def test_go2_connect_location_not_found(mock_go2_connector):
    """Test connect logs warning if dock location is not found."""
    mock_go2_connector.charging_provider.get_charging_status.return_value = 0
    mock_go2_connector.location_provider.get_location.return_value = None
    mock_go2_connector.location_provider.get_all_locations.return_value = {}

    inp = DockChargingInput(action="dock")
    await mock_go2_connector.connect(inp)

    mock_go2_connector.navigation_provider.publish_goal_pose.assert_not_called()


@pytest.mark.asyncio
async def test_go2_connect_navigates_to_dock(mock_go2_connector):
    """Test connect publishes navigation goal when dock location is found."""
    mock_go2_connector.charging_provider.get_charging_status.return_value = 0
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock")
    await mock_go2_connector.connect(inp)

    mock_go2_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_go2_connect_uses_custom_dock_location(mock_go2_connector):
    """Test connect uses dock_location_name from input over config default."""
    mock_go2_connector.charging_provider.get_charging_status.return_value = 0
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 3.0, "y": 4.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock", dock_location_name="dock2")
    await mock_go2_connector.connect(inp)

    mock_go2_connector.location_provider.get_location.assert_called_with("dock2")
    mock_go2_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_go2_connect_uses_config_default_when_no_location_in_input(
    mock_go2_connector,
):
    """Test connect falls back to config default_dock_location when input has None."""
    mock_go2_connector.charging_provider.get_charging_status.return_value = 0
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 1.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock", dock_location_name=None)
    await mock_go2_connector.connect(inp)

    mock_go2_connector.location_provider.get_location.assert_called_with(
        "charging_dock"
    )


@pytest.mark.asyncio
async def test_go2_connect_handles_navigation_exception(mock_go2_connector):
    """Test connect handles exception from navigation provider gracefully."""
    mock_go2_connector.charging_provider.get_charging_status.return_value = 0
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }
    mock_go2_connector.navigation_provider.publish_goal_pose.side_effect = Exception(
        "Navigation error"
    )

    inp = DockChargingInput(action="dock")
    await mock_go2_connector.connect(inp)


# ---------------------------------------------------------------------------
# G1 config tests
# ---------------------------------------------------------------------------


def test_g1_connector_config_defaults():
    """Test UnitreeG1DockConfig default values."""
    config = UnitreeG1DockConfig()
    assert config.base_url == "http://localhost:5000/maps/locations/list"
    assert config.timeout == 5
    assert config.refresh_interval == 30
    assert config.default_dock_location == "charging_dock"


def test_g1_connector_config_custom():
    """Test UnitreeG1DockConfig with custom values."""
    config = UnitreeG1DockConfig(
        base_url="http://192.168.1.1:5000/maps/locations/list",
        timeout=10,
        refresh_interval=60,
        default_dock_location="my_dock",
    )
    assert config.base_url == "http://192.168.1.1:5000/maps/locations/list"
    assert config.timeout == 10
    assert config.refresh_interval == 60
    assert config.default_dock_location == "my_dock"


# ---------------------------------------------------------------------------
# G1 connector tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_g1_connector():
    """Create a UnitreeG1DockConnector with all providers mocked."""
    with (
        patch(
            "actions.dock_charging.connector.unitree_g1_dock.UnitreeG1LocationsProvider"
        ) as mock_loc,
        patch(
            "actions.dock_charging.connector.unitree_g1_dock.UnitreeG1NavigationProvider"
        ) as mock_nav,
    ):
        config = UnitreeG1DockConfig()
        connector = UnitreeG1DockConnector(config=config)
        connector.location_provider = mock_loc.return_value
        connector.navigation_provider = mock_nav.return_value
        yield connector


@pytest.mark.asyncio
async def test_g1_connect_location_not_found(mock_g1_connector):
    """Test connect logs warning if dock location is not found."""
    mock_g1_connector.location_provider.get_location.return_value = None
    mock_g1_connector.location_provider.get_all_locations.return_value = {}

    inp = DockChargingInput(action="dock")
    await mock_g1_connector.connect(inp)

    mock_g1_connector.navigation_provider.publish_goal_pose.assert_not_called()


@pytest.mark.asyncio
async def test_g1_connect_navigates_to_dock(mock_g1_connector):
    """Test connect publishes navigation goal when dock location is found."""
    mock_g1_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock")
    await mock_g1_connector.connect(inp)

    mock_g1_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_g1_connect_uses_custom_dock_location(mock_g1_connector):
    """Test connect uses dock_location_name from input over config default."""
    mock_g1_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 3.0, "y": 4.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock", dock_location_name="dock2")
    await mock_g1_connector.connect(inp)

    mock_g1_connector.location_provider.get_location.assert_called_with("dock2")
    mock_g1_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_g1_connect_uses_config_default_when_no_location_in_input(
    mock_g1_connector,
):
    """Test connect falls back to config default_dock_location when input has None."""
    mock_g1_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 1.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = DockChargingInput(action="dock", dock_location_name=None)
    await mock_g1_connector.connect(inp)

    mock_g1_connector.location_provider.get_location.assert_called_with("charging_dock")


@pytest.mark.asyncio
async def test_g1_connect_handles_navigation_exception(mock_g1_connector):
    """Test connect handles exception from navigation provider gracefully."""
    mock_g1_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }
    mock_g1_connector.navigation_provider.publish_goal_pose.side_effect = Exception(
        "Navigation error"
    )

    inp = DockChargingInput(action="dock")
    await mock_g1_connector.connect(inp)
