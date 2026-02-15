from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from actions.base import ActionConfig
from actions.move_to_peer.connector.ros2 import MoveToPeerRos2Connector
from actions.move_to_peer.interface import MoveToPeerAction, MoveToPeerInput

_mock_unitree = MagicMock()
_mock_sport_client_class = Mock()
_mock_unitree.unitree_sdk2py.go2.sport.sport_client.SportClient = (
    _mock_sport_client_class
)
_unitree_mocks = {
    "unitree": _mock_unitree,
    "unitree.unitree_sdk2py": _mock_unitree.unitree_sdk2py,
    "unitree.unitree_sdk2py.go2": _mock_unitree.unitree_sdk2py.go2,
    "unitree.unitree_sdk2py.go2.sport": _mock_unitree.unitree_sdk2py.go2.sport,
    "unitree.unitree_sdk2py.go2.sport.sport_client": (
        _mock_unitree.unitree_sdk2py.go2.sport.sport_client
    ),
}


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""
    with (
        patch.dict("sys.modules", _unitree_mocks),
        patch("actions.move_to_peer.connector.ros2.IOProvider") as mock_io_class,
    ):
        mock_io = Mock()
        mock_io_class.return_value = mock_io

        mock_sport = Mock()
        _mock_sport_client_class.return_value = mock_sport

        config = ActionConfig()
        connector = MoveToPeerRos2Connector(config)

        yield connector, mock_io, mock_sport


class TestMoveToPeerRos2ConnectorInit:
    """Test MoveToPeerRos2Connector initialization."""

    def test_init(self, mock_dependencies):
        """Test successful initialization."""
        connector, mock_io, mock_sport = mock_dependencies
        assert connector.MAX_ROT_SPEED == 0.2
        assert connector.FWD_SPEED == 0.4
        assert connector.ANG_TOL_DEG == 5.0
        assert connector.STOP_DIST == 4.0
        mock_sport.SetTimeout.assert_called_once_with(10.0)
        mock_sport.Init.assert_called_once()


class TestMoveToPeerRos2ConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_idle(self, mock_dependencies):
        """Test idle action returns without movement."""
        connector, mock_io, mock_sport = mock_dependencies
        move_input = MoveToPeerInput(action=MoveToPeerAction.IDLE)
        with patch("actions.move_to_peer.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call(
                "MoveToPeer: idle, no movement commanded."
            )

    @pytest.mark.asyncio
    async def test_connect_no_own_location(self, mock_dependencies):
        """Test navigate when own GPS location is not available."""
        connector, mock_io, mock_sport = mock_dependencies
        mock_io.get_dynamic_variable.return_value = None
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        with patch("actions.move_to_peer.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call(
                "MoveToPeer: own location not available, not moving."
            )

    @pytest.mark.asyncio
    async def test_connect_no_peer_location(self, mock_dependencies):
        """Test navigate when peer GPS location is not available."""
        connector, mock_io, mock_sport = mock_dependencies

        def get_var(name):
            if name in ("latitude", "longitude"):
                return 40.0
            return None

        mock_io.get_dynamic_variable.side_effect = get_var
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        with patch("actions.move_to_peer.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call(
                "MoveToPeer: peer location not available, not moving."
            )

    @pytest.mark.asyncio
    async def test_connect_already_near_peer(self, mock_dependencies):
        """Test navigate when already near the peer."""
        connector, mock_io, mock_sport = mock_dependencies

        def get_var(name):
            mapping = {
                "latitude": 40.0,
                "longitude": -74.0,
                "closest_peer_lat": 40.0,
                "closest_peer_lon": -74.0,
                "yaw_deg": 0.0,
            }
            return mapping.get(name)

        mock_io.get_dynamic_variable.side_effect = get_var
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        with patch("actions.move_to_peer.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call(
                "MoveToPeer: already near peer (d=0.0\u00a0m < 4.0\u00a0m)."
            )

    @pytest.mark.asyncio
    async def test_connect_no_yaw_drives_body_frame(self, mock_dependencies):
        """Test navigate without yaw uses body-frame vector."""
        connector, mock_io, mock_sport = mock_dependencies
        mock_sport.reset_mock()

        def get_var(name):
            mapping = {
                "latitude": 40.0,
                "longitude": -74.0,
                "closest_peer_lat": 40.001,
                "closest_peer_lon": -74.0,
                "yaw_deg": None,
            }
            return mapping.get(name)

        mock_io.get_dynamic_variable.side_effect = get_var
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        with patch("actions.move_to_peer.connector.ros2.logging") as mock_logging:
            await connector.connect(move_input)
            mock_logging.info.assert_any_call(
                "MoveToPeer: yaw unknown \u2192 driving body\u2011frame vector instead."
            )
            mock_sport.Move.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_aligned_drives_forward(self, mock_dependencies):
        """Test navigate when already aligned drives forward."""
        connector, mock_io, mock_sport = mock_dependencies
        mock_sport.reset_mock()

        def get_var(name):
            mapping = {
                "latitude": 40.0,
                "longitude": -74.0,
                "closest_peer_lat": 40.001,
                "closest_peer_lon": -74.0,
                "yaw_deg": 0.0,
            }
            return mapping.get(name)

        mock_io.get_dynamic_variable.side_effect = get_var
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        await connector.connect(move_input)
        mock_sport.Move.assert_called_once_with(0.4, 0.0, 0.0)

    @pytest.mark.asyncio
    async def test_connect_misaligned_rotates(self, mock_dependencies):
        """Test navigate when misaligned rotates in place."""
        connector, mock_io, mock_sport = mock_dependencies
        mock_sport.reset_mock()

        def get_var(name):
            mapping = {
                "latitude": 40.0,
                "longitude": -74.0,
                "closest_peer_lat": 40.0,
                "closest_peer_lon": -73.999,  # East of current
                "yaw_deg": 180.0,  # Facing south
            }
            return mapping.get(name)

        mock_io.get_dynamic_variable.side_effect = get_var
        move_input = MoveToPeerInput(action=MoveToPeerAction.NAVIGATE)
        with patch("actions.move_to_peer.connector.ros2.asyncio") as mock_asyncio:
            mock_asyncio.sleep = AsyncMock()
            await connector.connect(move_input)
            mock_sport.Move.assert_called_once()
            call_args = mock_sport.Move.call_args[0]
            assert call_args[0] == 0.0  # vx should be 0 (rotating in place)
