from unittest.mock import patch

import pytest

from actions.return_to_base.connector.unitree_g1_return import (
    UnitreeG1ReturnToBaseConfig,
    UnitreeG1ReturnToBaseConnector,
)
from actions.return_to_base.connector.unitree_go2_return import (
    UnitreeGo2ReturnToBaseConfig,
    UnitreeGo2ReturnToBaseConnector,
)
from actions.return_to_base.interface import ReturnToBase, ReturnToBaseInput


def test_return_to_base_input_defaults():
    """Test ReturnToBaseInput default values."""
    inp = ReturnToBaseInput(action="return")
    assert inp.action == "return"
    assert inp.base_location_name == "base"


def test_return_to_base_input_custom_location():
    """Test ReturnToBaseInput with custom base location name."""
    inp = ReturnToBaseInput(action="return", base_location_name="home_base")
    assert inp.base_location_name == "home_base"


def test_return_to_base_interface():
    """Test ReturnToBase interface structure."""
    inp = ReturnToBaseInput(action="return")
    interface = ReturnToBase(input=inp, output=inp)
    assert interface.input == inp
    assert interface.output == inp


# ── Go2 ────────────────────────────────────────────────────────────────────────


def test_go2_config_defaults():
    """Test UnitreeGo2ReturnToBaseConfig default values."""
    config = UnitreeGo2ReturnToBaseConfig()
    assert config.base_url == "http://localhost:5000/maps/locations/list"
    assert config.timeout == 5
    assert config.refresh_interval == 30
    assert config.default_base_location == "base"


def test_go2_config_custom():
    """Test UnitreeGo2ReturnToBaseConfig with custom values."""
    config = UnitreeGo2ReturnToBaseConfig(
        base_url="http://192.168.1.1:5000/maps/locations/list",
        timeout=10,
        refresh_interval=60,
        default_base_location="home",
    )
    assert config.base_url == "http://192.168.1.1:5000/maps/locations/list"
    assert config.timeout == 10
    assert config.refresh_interval == 60
    assert config.default_base_location == "home"


@pytest.fixture
def mock_go2_connector():
    """Create a UnitreeGo2ReturnToBaseConnector with all providers mocked."""
    with (
        patch(
            "actions.return_to_base.connector.unitree_go2_return.UnitreeGo2LocationsProvider"
        ) as mock_loc,
        patch(
            "actions.return_to_base.connector.unitree_go2_return.UnitreeGo2NavigationProvider"
        ) as mock_nav,
    ):
        config = UnitreeGo2ReturnToBaseConfig()
        connector = UnitreeGo2ReturnToBaseConnector(config=config)
        connector.location_provider = mock_loc.return_value
        connector.navigation_provider = mock_nav.return_value
        yield connector


@pytest.mark.asyncio
async def test_go2_location_not_found(mock_go2_connector):
    """Test connect logs warning if base location is not found."""
    mock_go2_connector.location_provider.get_location.return_value = None
    mock_go2_connector.location_provider.get_all_locations.return_value = {}

    inp = ReturnToBaseInput(action="return")
    await mock_go2_connector.connect(inp)

    mock_go2_connector.navigation_provider.publish_goal_pose.assert_not_called()


@pytest.mark.asyncio
async def test_go2_navigates_to_base(mock_go2_connector):
    """Test connect publishes navigation goal when base location is found."""
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = ReturnToBaseInput(action="return")
    await mock_go2_connector.connect(inp)

    mock_go2_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_go2_uses_custom_base_location(mock_go2_connector):
    """Test connect uses base_location_name from input over config default."""
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 3.0, "y": 4.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = ReturnToBaseInput(action="return", base_location_name="home_base")
    await mock_go2_connector.connect(inp)

    mock_go2_connector.location_provider.get_location.assert_called_with("home_base")
    mock_go2_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_go2_uses_config_default_when_no_location_in_input(mock_go2_connector):
    """Test connect falls back to config default_base_location when input has None."""
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 1.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = ReturnToBaseInput(action="return", base_location_name=None)
    await mock_go2_connector.connect(inp)

    mock_go2_connector.location_provider.get_location.assert_called_with("base")


@pytest.mark.asyncio
async def test_go2_handles_navigation_exception(mock_go2_connector):
    """Test connect handles exception from navigation provider gracefully."""
    mock_go2_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }
    mock_go2_connector.navigation_provider.publish_goal_pose.side_effect = Exception(
        "Navigation error"
    )

    inp = ReturnToBaseInput(action="return")
    await mock_go2_connector.connect(inp)


# ── G1 ─────────────────────────────────────────────────────────────────────────


def test_g1_config_defaults():
    """Test UnitreeG1ReturnToBaseConfig default values."""
    config = UnitreeG1ReturnToBaseConfig()
    assert config.base_url == "http://localhost:5000/maps/locations/list"
    assert config.timeout == 5
    assert config.refresh_interval == 30
    assert config.default_base_location == "base"


def test_g1_config_custom():
    """Test UnitreeG1ReturnToBaseConfig with custom values."""
    config = UnitreeG1ReturnToBaseConfig(
        base_url="http://192.168.1.1:5000/maps/locations/list",
        timeout=10,
        refresh_interval=60,
        default_base_location="home",
    )
    assert config.base_url == "http://192.168.1.1:5000/maps/locations/list"
    assert config.timeout == 10
    assert config.refresh_interval == 60
    assert config.default_base_location == "home"


@pytest.fixture
def mock_g1_connector():
    """Create a UnitreeG1ReturnToBaseConnector with all providers mocked."""
    with (
        patch(
            "actions.return_to_base.connector.unitree_g1_return.UnitreeG1LocationsProvider"
        ) as mock_loc,
        patch(
            "actions.return_to_base.connector.unitree_g1_return.UnitreeG1NavigationProvider"
        ) as mock_nav,
    ):
        config = UnitreeG1ReturnToBaseConfig()
        connector = UnitreeG1ReturnToBaseConnector(config=config)
        connector.location_provider = mock_loc.return_value
        connector.navigation_provider = mock_nav.return_value
        yield connector


@pytest.mark.asyncio
async def test_g1_location_not_found(mock_g1_connector):
    """Test connect logs warning if base location is not found."""
    mock_g1_connector.location_provider.get_location.return_value = None
    mock_g1_connector.location_provider.get_all_locations.return_value = {}

    inp = ReturnToBaseInput(action="return")
    await mock_g1_connector.connect(inp)

    mock_g1_connector.navigation_provider.publish_goal_pose.assert_not_called()


@pytest.mark.asyncio
async def test_g1_navigates_to_base(mock_g1_connector):
    """Test connect publishes navigation goal when base location is found."""
    mock_g1_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = ReturnToBaseInput(action="return")
    await mock_g1_connector.connect(inp)

    mock_g1_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_g1_uses_custom_base_location(mock_g1_connector):
    """Test connect uses base_location_name from input over config default."""
    mock_g1_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 3.0, "y": 4.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = ReturnToBaseInput(action="return", base_location_name="home_base")
    await mock_g1_connector.connect(inp)

    mock_g1_connector.location_provider.get_location.assert_called_with("home_base")
    mock_g1_connector.navigation_provider.publish_goal_pose.assert_called_once()


@pytest.mark.asyncio
async def test_g1_uses_config_default_when_no_location_in_input(mock_g1_connector):
    """Test connect falls back to config default_base_location when input has None."""
    mock_g1_connector.location_provider.get_location.return_value = {
        "pose": {
            "position": {"x": 1.0, "y": 1.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    }

    inp = ReturnToBaseInput(action="return", base_location_name=None)
    await mock_g1_connector.connect(inp)

    mock_g1_connector.location_provider.get_location.assert_called_with("base")


@pytest.mark.asyncio
async def test_g1_handles_navigation_exception(mock_g1_connector):
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

    inp = ReturnToBaseInput(action="return")
    await mock_g1_connector.connect(inp)
