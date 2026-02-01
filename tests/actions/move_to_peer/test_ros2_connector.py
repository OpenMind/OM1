"""Tests for MoveToPeerRos2Connector validation and early-return logic."""

import sys
import types
from unittest.mock import Mock

import pytest

from actions.base import ActionConfig
from actions.move_to_peer.interface import MoveToPeerAction, MoveToPeerInput

# Stub providers/unitree so importing the connector does not load zenoh/unitree SDK.
# Save/restore in fixture so other tests in the session are not affected.
_saved_modules = {}
_mock_io_module = types.ModuleType("io_provider")
_mock_io_module.IOProvider = Mock()
_unitree_sport_mod = None


def _install_stubs():
    global _unitree_sport_mod
    if "providers" not in sys.modules:
        sys.modules["providers"] = types.ModuleType("providers")
    _saved_modules["providers.io_provider"] = sys.modules.get("providers.io_provider")
    sys.modules["providers.io_provider"] = _mock_io_module
    _saved_modules["unitree.unitree_sdk2py.go2.sport.sport_client"] = sys.modules.get(
        "unitree.unitree_sdk2py.go2.sport.sport_client"
    )
    for _p in (
        "unitree",
        "unitree.unitree_sdk2py",
        "unitree.unitree_sdk2py.go2",
        "unitree.unitree_sdk2py.go2.sport",
        "unitree.unitree_sdk2py.go2.sport.sport_client",
    ):
        if _p not in sys.modules:
            sys.modules[_p] = types.ModuleType(_p.split(".")[-1])
    _unitree_sport_mod = sys.modules["unitree.unitree_sdk2py.go2.sport.sport_client"]
    _unitree_sport_mod.SportClient = Mock()


def _restore_modules():
    for name, mod in _saved_modules.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod
    _saved_modules.clear()


_install_stubs()
from actions.move_to_peer.connector.ros2 import MoveToPeerRos2Connector


@pytest.fixture(scope="module", autouse=True)
def _restore_sys_modules_after_move_to_peer_connector_tests():
    """Restore sys.modules after all tests so other test modules are not affected."""
    yield
    _restore_modules()


@pytest.fixture
def mock_io_and_sport():
    """Mock IOProvider and SportClient instances for connector."""
    mock_io = Mock()
    mock_sport = Mock()
    _mock_io_module.IOProvider.return_value = mock_io
    # Use stub module ref (ros2 still has SportClient = this Mock after restore)
    _unitree_sport_mod.SportClient.return_value = mock_sport
    yield mock_io, mock_sport
    _mock_io_module.IOProvider.return_value = None
    _unitree_sport_mod.SportClient.return_value = None


@pytest.fixture
def connector(mock_io_and_sport):
    """Create MoveToPeerRos2Connector with mocked IOProvider and SportClient."""
    mock_io, mock_sport = mock_io_and_sport
    return MoveToPeerRos2Connector(ActionConfig())


@pytest.mark.asyncio
async def test_connect_idle_returns_without_moving(connector, mock_io_and_sport):
    """When action is IDLE, connect returns without calling sport_client.Move."""
    _mock_io, mock_sport = mock_io_and_sport
    inp = MoveToPeerInput(action=MoveToPeerAction.IDLE)
    await connector.connect(inp)
    mock_sport.Move.assert_not_called()


@pytest.mark.asyncio
async def test_connect_own_location_missing_returns_without_moving(
    connector, mock_io_and_sport
):
    """When own lat/lon is missing, connect returns without moving."""
    mock_io, mock_sport = mock_io_and_sport
    mock_io.get_dynamic_variable.side_effect = lambda key: {
        "latitude": None,
        "longitude": None,
        "closest_peer_lat": "50.0",
        "closest_peer_lon": "10.0",
        "yaw_deg": "0.0",
    }.get(key)
    inp = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
    await connector.connect(inp)
    mock_sport.Move.assert_not_called()


@pytest.mark.asyncio
async def test_connect_peer_location_missing_returns_without_moving(
    connector, mock_io_and_sport
):
    """When peer lat/lon is missing, connect returns without moving."""
    mock_io, mock_sport = mock_io_and_sport
    mock_io.get_dynamic_variable.side_effect = lambda key: {
        "latitude": "50.0",
        "longitude": "10.0",
        "closest_peer_lat": None,
        "closest_peer_lon": None,
        "yaw_deg": "0.0",
    }.get(key)
    inp = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
    await connector.connect(inp)
    mock_sport.Move.assert_not_called()


@pytest.mark.asyncio
async def test_connect_already_near_peer_returns_without_moving(
    connector, mock_io_and_sport
):
    """When distance to peer is below STOP_DIST, connect returns without moving."""
    mock_io, mock_sport = mock_io_and_sport
    # Two points ~1 m apart (STOP_DIST is 4 m)
    mock_io.get_dynamic_variable.side_effect = lambda key: {
        "latitude": "50.0",
        "longitude": "10.0",
        "closest_peer_lat": "50.00001",
        "closest_peer_lon": "10.0",
        "yaw_deg": "0.0",
    }.get(key)
    inp = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
    await connector.connect(inp)
    mock_sport.Move.assert_not_called()
