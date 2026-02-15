from unittest.mock import AsyncMock, Mock, patch

import pytest

from llm.output_model import Action
from src.simulators.plugins.WebSim import SimulatorConfig, WebSim


class MockInput:
    def __init__(self, input_type, input, timestamp):
        self.input_type = input_type
        self.input = input
        self.timestamp = timestamp


class TestWebSim:
    @pytest.fixture
    def mock_fastapi(self):
        with (
            patch("src.simulators.plugins.WebSim.FastAPI") as mock_fastapi,
            patch("src.simulators.plugins.WebSim.StaticFiles") as mock_static,
            patch("src.simulators.plugins.WebSim.uvicorn") as mock_uvicorn,
        ):
            mock_app = Mock()
            mock_fastapi.return_value = mock_app
            yield mock_fastapi, mock_static, mock_uvicorn, mock_app

    @pytest.fixture
    def config(self):
        return SimulatorConfig(name="WebSim")

    @pytest.fixture
    def websim(self, mock_fastapi, config):
        with (
            patch("src.simulators.plugins.WebSim.threading.Thread") as mock_thread,
            patch("src.simulators.plugins.WebSim.os.path.exists") as mock_exists,
        ):
            mock_exists.return_value = True
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            mock_thread_instance.is_alive.return_value = True
            ws = WebSim(config)
            return ws

    def test_init_server_thread_started(self, websim, mock_fastapi):
        """Test that server thread is started and WebSim is initialized."""
        mock_fastapi, mock_static, mock_uvicorn, mock_app = mock_fastapi
        assert websim._initialized is True
        mock_fastapi.assert_called_once()
        mock_app.mount.assert_called()

    def test_get_earliest_time(self, websim):
        """Test get_earliest_time returns correct minimum timestamp."""
        inputs = {
            "Camera": MockInput(input_type="Camera", input="image", timestamp=10.0),
            "Voice": MockInput(input_type="Voice", input="hello", timestamp=5.0),
            "GovernanceEthereum": MockInput(
                input_type="GovernanceEthereum", input="", timestamp=100.0
            ),
            "Universal Laws": MockInput(
                input_type="Universal Laws", input="", timestamp=200.0
            ),
        }
        assert websim.get_earliest_time(inputs) == 5.0

    def test_get_earliest_time_no_valid(self, websim):
        """Test when no valid inputs, return 0.0."""
        inputs = {
            "GovernanceEthereum": MockInput(
                input_type="GovernanceEthereum", input="", timestamp=100.0
            ),
        }
        assert websim.get_earliest_time(inputs) == 0.0

    @pytest.mark.asyncio
    async def test_broadcast_state(self, websim):
        """Test broadcast_state sends data to all active connections."""
        mock_connection1 = AsyncMock()
        mock_connection2 = AsyncMock()
        websim.active_connections = [mock_connection1, mock_connection2]
        websim.state_dict = {"test": "data"}

        await websim.broadcast_state()

        mock_connection1.send_json.assert_called_once_with({"test": "data"})
        mock_connection2.send_json.assert_called_once_with({"test": "data"})

    @pytest.mark.asyncio
    async def test_broadcast_state_with_disconnected(self, websim):
        """Test broadcast_state handles disconnected clients."""
        mock_connection1 = AsyncMock()
        mock_connection2 = AsyncMock()
        mock_connection1.send_json.side_effect = Exception("disconnected")
        websim.active_connections = [mock_connection1, mock_connection2]
        websim.state_dict = {"test": "data"}

        await websim.broadcast_state()

        # connection1 should be removed
        assert mock_connection1 not in websim.active_connections
        assert mock_connection2 in websim.active_connections
        mock_connection2.send_json.assert_called_once()

    def test_tick_calls_broadcast_state(self, websim):
        """Test tick runs broadcast_state in event loop."""
        websim._initialized = True
        with patch.object(
            websim, "broadcast_state", new_callable=AsyncMock
        ) as mock_broadcast:
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_get_loop.return_value = mock_loop
                mock_loop.is_running.return_value = False

                websim.tick()

                mock_loop.run_until_complete.assert_called_once()
                mock_broadcast.assert_called_once()

    @pytest.mark.filterwarnings(
        "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited"
    )
    def test_tick_not_initialized(self, websim):
        """Test tick does nothing if not initialized."""
        websim._initialized = False
        with patch.object(
            websim, "broadcast_state", new_callable=AsyncMock
        ) as mock_broadcast:
            websim.tick()
            mock_broadcast.assert_not_called()

    def test_sim_updates_state(self, websim):
        """Test sim processes actions and updates state."""
        websim._initialized = True
        websim.io_provider = Mock()
        websim.io_provider.inputs = {
            "Camera": MockInput(input_type="Camera", input="image", timestamp=10.0),
            "Voice": MockInput(input_type="Voice", input="hello", timestamp=5.0),
        }
        websim.io_provider.fuser_start_time = 100.0
        websim.io_provider.fuser_end_time = 102.0
        websim.io_provider.llm_start_time = 101.0
        websim.io_provider.llm_end_time = 103.0

        actions = [
            Action(type="move", value="walk"),
            Action(type="speak", value="Hello world"),
            Action(type="emotion", value="happy"),
        ]

        with patch.object(websim, "tick") as mock_tick:
            websim.sim(actions)

        assert websim.state.current_action == "walk"
        assert websim.state.last_speech == "Hello world"
        assert websim.state.current_emotion == "happy"
        assert websim.state_dict["current_action"] == "walk"
        assert websim.state_dict["last_speech"] == "Hello world"
        assert websim.state_dict["current_emotion"] == "happy"
        assert "inputs" in websim.state_dict
        mock_tick.assert_called_once()

    def test_sim_not_initialized(self, websim):
        """Test sim does nothing if not initialized."""
        websim._initialized = False
        websim.sim([Action(type="move", value="walk")])
        assert websim.state.current_action == "idle"
