import time
from unittest.mock import Mock, patch

import pytest

from actions.explore.connector.unitree_go2_explore import (
    UnitreeGo2ExploreConfig,
    UnitreeGo2ExploreConnector,
)
from actions.explore.interface import ExploreInput


class TestUnitreeGo2ExploreConfig:
    def test_default_config(self):
        config = UnitreeGo2ExploreConfig()
        assert config.explore_start_topic == "explore/start"
        assert config.explore_stop_topic == "explore/stop"
        assert config.unitree_ethernet == "eth0"
        assert config.return_speed == 0.5

    def test_custom_config(self):
        config = UnitreeGo2ExploreConfig(
            explore_start_topic="my/start",
            explore_stop_topic="my/stop",
            unitree_ethernet="enP2p1s0",
            return_speed=0.8,
        )
        assert config.explore_start_topic == "my/start"
        assert config.explore_stop_topic == "my/stop"
        assert config.unitree_ethernet == "enP2p1s0"
        assert config.return_speed == 0.8


class TestUnitreeGo2ExploreConnectorInit:
    def test_init_providers_called(self):
        with (
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2OdomProvider"
            ) as mock_odom,
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2FrontierExplorationProvider"
            ) as mock_frontier,
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2NavigationProvider"
            ) as mock_nav,
            patch(
                "actions.explore.connector.unitree_go2_explore.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("actions.explore.connector.unitree_go2_explore.open_zenoh_session"),
        ):
            config = UnitreeGo2ExploreConfig()
            connector = UnitreeGo2ExploreConnector(config)

            mock_odom.assert_called_once_with(channel="eth0")
            mock_frontier.assert_called_once()
            mock_frontier.return_value.start.assert_called_once()
            mock_nav.assert_called_once()
            mock_nav.return_value.start.assert_called_once()
            mock_tts.assert_called_once()

            assert connector._exploring is False
            assert connector._start_position is None
            assert connector._start_time is None
            assert connector._duration is None
            assert connector._return_to_start is True

    def test_init_zenoh_session_failure(self):
        with (
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2FrontierExplorationProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2NavigationProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.ElevenLabsTTSProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.open_zenoh_session",
                side_effect=Exception("Connection refused"),
            ),
        ):
            config = UnitreeGo2ExploreConfig()
            connector = UnitreeGo2ExploreConnector(config)
            assert connector.session is None
            assert connector._start_pub is None
            assert connector._stop_pub is None


class TestUnitreeGo2ExploreConnectorConnect:
    @pytest.fixture
    def connector(self):
        with (
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2OdomProvider"
            ) as mock_odom,
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2FrontierExplorationProvider"
            ) as mock_frontier,
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2NavigationProvider"
            ) as mock_nav,
            patch(
                "actions.explore.connector.unitree_go2_explore.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("actions.explore.connector.unitree_go2_explore.open_zenoh_session"),
        ):
            mock_odom_instance = Mock()
            mock_odom_instance.position = {
                "odom_x": 1.0,
                "odom_y": 2.0,
                "odom_yaw_m180_p180": 0.5,
            }
            mock_odom.return_value = mock_odom_instance

            mock_frontier_instance = Mock()
            mock_frontier_instance.status = False
            mock_frontier_instance.exploration_complete = False
            mock_frontier.return_value = mock_frontier_instance

            mock_nav_instance = Mock()
            mock_nav.return_value = mock_nav_instance

            mock_tts_instance = Mock()
            mock_tts.return_value = mock_tts_instance
            config = UnitreeGo2ExploreConfig()
            conn = UnitreeGo2ExploreConnector(config)
            yield conn, mock_odom_instance, mock_frontier_instance, mock_nav_instance, mock_tts_instance

    @pytest.mark.asyncio
    async def test_connect_explore_starts(self, connector):
        conn, _, _, _, mock_tts = connector
        with patch.object(conn, "_publish") as mock_pub:
            await conn.connect(ExploreInput(action="explore"))
            assert conn._exploring is True
            assert conn._start_position == (1.0, 2.0, 0.5)
            assert conn._start_time is not None
            mock_pub.assert_called_once()
            mock_tts.add_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_explore_with_duration(self, connector):
        conn, *_ = connector
        with patch.object(conn, "_publish"):
            await conn.connect(ExploreInput(action="explore", duration=60))
        assert conn._duration == 60

    @pytest.mark.asyncio
    async def test_connect_explore_return_to_start_false(self, connector):
        conn, *_ = connector
        with patch.object(conn, "_publish"):
            await conn.connect(ExploreInput(action="explore", return_to_start=False))
        assert conn._return_to_start is False

    @pytest.mark.asyncio
    async def test_connect_explore_already_exploring(self, connector):
        conn, *_ = connector
        conn._exploring = True
        with patch.object(conn, "_publish") as mock_pub:
            await conn.connect(ExploreInput(action="explore"))
            mock_pub.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_stop_explore(self, connector):
        conn, *_ = connector
        conn._exploring = True
        with patch.object(conn, "_stop_exploration") as mock_stop:
            await conn.connect(ExploreInput(action="stop explore"))
            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_stop_not_exploring(self, connector):
        conn, *_ = connector
        conn._exploring = False
        with patch.object(conn, "_stop_exploration") as mock_stop:
            await conn.connect(ExploreInput(action="stop explore"))
            mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_unknown_action(self, connector):
        conn, *_ = connector
        with patch.object(conn, "_publish") as mock_pub:
            await conn.connect(ExploreInput(action="fly away"))
            mock_pub.assert_not_called()
            assert conn._exploring is False

    @pytest.mark.asyncio
    async def test_connect_odom_not_available(self, connector):
        conn, mock_odom, *_ = connector
        mock_odom.position = None
        with patch.object(conn, "_publish"):
            await conn.connect(ExploreInput(action="explore"))
        assert conn._exploring is True
        assert conn._start_position is None


class TestUnitreeGo2ExploreConnectorTick:
    @pytest.fixture
    def connector(self):
        with (
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2FrontierExplorationProvider"
            ) as mock_frontier,
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2NavigationProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("actions.explore.connector.unitree_go2_explore.open_zenoh_session"),
        ):
            mock_frontier_instance = Mock()
            mock_frontier_instance.status = False
            mock_frontier.return_value = mock_frontier_instance

            mock_tts_instance = Mock()
            mock_tts.return_value = mock_tts_instance
            config = UnitreeGo2ExploreConfig()
            conn = UnitreeGo2ExploreConnector(config)
            yield conn, mock_frontier_instance, mock_tts_instance

    def test_tick_not_exploring_sleeps(self, connector):
        conn, *_ = connector
        conn._exploring = False
        with patch.object(conn, "sleep") as mock_sleep:
            conn.tick()
            mock_sleep.assert_called_once_with(1.0)

    def test_tick_exploring_no_stop_condition(self, connector):
        conn, mock_frontier, _ = connector
        conn._exploring = True
        conn._start_time = time.time()
        conn._duration = 9999
        mock_frontier.status = False
        with (
            patch.object(conn, "sleep") as mock_sleep,
            patch.object(conn, "_stop_exploration") as mock_stop,
        ):
            conn.tick()
            mock_stop.assert_not_called()
            mock_sleep.assert_called_once_with(1.0)

    def test_tick_stops_on_duration_exceeded(self, connector):
        conn, mock_frontier, mock_tts = connector
        conn._exploring = True
        conn._start_time = time.time() - 100
        conn._duration = 10
        conn._return_to_start = False
        mock_frontier.status = False
        with patch.object(conn, "_stop_exploration") as mock_stop:
            conn.tick()
            mock_stop.assert_called_once()
            mock_tts.add_pending_message.assert_called_once()

    def test_tick_stops_on_exploration_complete(self, connector):
        conn, mock_frontier, _ = connector
        conn._exploring = True
        conn._start_time = time.time()
        conn._duration = None
        conn._return_to_start = False
        mock_frontier.status = True
        with patch.object(conn, "_stop_exploration") as mock_stop:
            conn.tick()
            mock_stop.assert_called_once()

    def test_tick_navigates_to_start_when_done(self, connector):
        conn, mock_frontier, _ = connector
        conn._exploring = True
        conn._start_time = time.time()
        conn._duration = None
        conn._return_to_start = True
        conn._start_position = (1.0, 2.0, 0.0)
        mock_frontier.status = True
        with (
            patch.object(conn, "_stop_exploration"),
            patch.object(conn, "_navigate_to_start") as mock_nav,
        ):
            conn.tick()
            mock_nav.assert_called_once()

    def test_tick_no_return_when_return_to_start_false(self, connector):
        conn, mock_frontier, _ = connector
        conn._exploring = True
        conn._start_time = time.time()
        conn._duration = None
        conn._return_to_start = False
        conn._start_position = (1.0, 2.0, 0.0)
        mock_frontier.status = True
        with (
            patch.object(conn, "_stop_exploration"),
            patch.object(conn, "_navigate_to_start") as mock_nav,
        ):
            conn.tick()
            mock_nav.assert_not_called()


class TestUnitreeGo2ExploreConnectorStop:
    @pytest.fixture
    def connector(self):
        with (
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2OdomProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2FrontierExplorationProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2NavigationProvider"
            ),
            patch(
                "actions.explore.connector.unitree_go2_explore.ElevenLabsTTSProvider"
            ),
            patch("actions.explore.connector.unitree_go2_explore.open_zenoh_session"),
        ):
            config = UnitreeGo2ExploreConfig()
            conn = UnitreeGo2ExploreConnector(config)
            yield conn

    def test_stop_calls_stop_exploration_if_exploring(self, connector):
        connector._exploring = True
        with patch.object(connector, "_stop_exploration") as mock_stop:
            connector.stop()
            mock_stop.assert_called_once()

    def test_stop_does_not_call_stop_exploration_if_idle(self, connector):
        connector._exploring = False
        with patch.object(connector, "_stop_exploration") as mock_stop:
            connector.stop()
            mock_stop.assert_not_called()

    def test_stop_closes_zenoh_session(self, connector):
        mock_session = Mock()
        connector.session = mock_session
        connector._exploring = False
        connector.stop()
        mock_session.close.assert_called_once()


class TestUnitreeGo2ExploreConnectorEdgeCases:
    @pytest.fixture
    def connector(self):
        with (
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2OdomProvider"
            ) as mock_odom,
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2FrontierExplorationProvider"
            ) as mock_frontier,
            patch(
                "actions.explore.connector.unitree_go2_explore.UnitreeGo2NavigationProvider"
            ) as mock_nav,
            patch(
                "actions.explore.connector.unitree_go2_explore.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch("actions.explore.connector.unitree_go2_explore.open_zenoh_session"),
        ):
            mock_odom_instance = Mock()
            mock_odom_instance.position = {
                "odom_x": 1.0,
                "odom_y": 2.0,
                "odom_yaw_m180_p180": 0.5,
            }
            mock_odom.return_value = mock_odom_instance

            mock_frontier_instance = Mock()
            mock_frontier_instance.status = False
            mock_frontier.return_value = mock_frontier_instance

            mock_nav_instance = Mock()
            mock_nav.return_value = mock_nav_instance

            mock_tts_instance = Mock()
            mock_tts.return_value = mock_tts_instance

            config = UnitreeGo2ExploreConfig()
            conn = UnitreeGo2ExploreConnector(config)
            yield conn, mock_odom_instance, mock_frontier_instance, mock_nav_instance, mock_tts_instance

    @pytest.mark.asyncio
    async def test_connect_odom_raises_exception(self, connector):
        conn, mock_odom, *_ = connector
        mock_odom.position = Mock(side_effect=Exception("odom crash"))
        with patch.object(conn, "_publish"):
            await conn.connect(ExploreInput(action="explore"))
        assert conn._exploring is True
        assert conn._start_position is None

    def test_navigate_to_start_no_position(self, connector):
        conn, *_ = connector
        conn._start_position = None
        conn._navigate_to_start()

    def test_navigate_to_start_happy_path(self, connector):
        conn, _, _, mock_nav, mock_tts = connector
        conn._start_position = (1.0, 2.0, 0.5)
        conn._navigate_to_start()
        mock_nav.publish_goal_pose.assert_called_once()
        mock_tts.add_pending_message.assert_called()

    def test_navigate_to_start_exception(self, connector):
        conn, _, _, mock_nav, _ = connector
        conn._start_position = (1.0, 2.0, 0.5)
        mock_nav.publish_goal_pose.side_effect = Exception("nav crash")
        conn._navigate_to_start()

    def test_publish_publisher_none(self, connector):
        conn, *_ = connector
        conn._publish(None, b"test", "some/topic")

    def test_publish_raises_exception(self, connector):
        conn, *_ = connector
        mock_pub = Mock()
        mock_pub.put.side_effect = Exception("zenoh error")
        conn._publish(mock_pub, b"test", "some/topic")
        mock_pub.put.assert_called_once()

    def test_stop_undeclare_raises_exception(self, connector):
        conn, *_ = connector
        conn._exploring = False
        mock_pub = Mock()
        mock_pub.undeclare.side_effect = Exception("undeclare error")
        conn._start_pub = mock_pub
        conn._stop_pub = mock_pub
        conn.stop()

    def test_stop_session_close_raises_exception(self, connector):
        conn, *_ = connector
        conn._exploring = False
        mock_session = Mock()
        mock_session.close.side_effect = Exception("close error")
        conn.session = mock_session
        conn.stop()

    def test_stop_exploration_real_call(self, connector):
        conn, *_ = connector
        conn._exploring = True
        mock_pub = Mock()
        conn._stop_pub = mock_pub
        conn._stop_exploration()
        assert conn._exploring is False
        mock_pub.put.assert_called_once()

    def test_publish_happy_path_real_call(self, connector):
        conn, *_ = connector
        mock_pub = Mock()
        conn._publish(mock_pub, b"hello", "test/topic")
        mock_pub.put.assert_called_once()
