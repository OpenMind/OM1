from unittest.mock import MagicMock, patch

import pytest
import zenoh

from zenoh_msgs.session import (
    HybridZenohSession,
    _create_zenoh_session,
    _open_cloud_session,
    _ZenohSessionContextManager,
    create_zenoh_config,
    load_api_key,
    load_session_config,
    load_use_sim,
    open_zenoh_session,
)


class TestCreateZenohConfig:
    def test_create_config_with_network_discovery_enabled(self):
        config = create_zenoh_config()
        assert isinstance(config, zenoh.Config)

    def test_create_config_with_network_discovery_disabled(self):
        config = create_zenoh_config(network_discovery=False)
        assert isinstance(config, zenoh.Config)


class TestOpenCloudSession:
    """Test the _open_cloud_session function."""

    @patch("zenoh_msgs.session.CloudSimZenohSession")
    def test_open_cloud_session_with_default_url(self, mock_cloud_session_class):
        mock_session = MagicMock()
        mock_cloud_session_class.return_value = mock_session

        result = _open_cloud_session()

        mock_cloud_session_class.assert_called_once_with("wss://api.openmind.com/api/core/simulation/zenoh", token=None)
        assert result is mock_session

    @patch("zenoh_msgs.session.CloudSimZenohSession")
    def test_open_cloud_session_with_custom_url_and_token(self, mock_cloud_session_class):
        mock_session = MagicMock()
        mock_cloud_session_class.return_value = mock_session

        result = _open_cloud_session(url="wss://custom.url:8444", token="my_token")

        mock_cloud_session_class.assert_called_once_with("wss://custom.url:8444", token="my_token")
        assert result is mock_session


class TestOpenZenohSession:
    """Test open_zenoh_session returns a context manager."""

    def test_returns_context_manager(self):
        result = open_zenoh_session()
        assert isinstance(result, _ZenohSessionContextManager)


class TestZenohSessionContextManager:
    """Test the _ZenohSessionContextManager class."""

    @patch("zenoh_msgs.session._create_zenoh_session")
    def test_enter_creates_session(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        ctx_mgr = _ZenohSessionContextManager()
        session = ctx_mgr.__enter__()

        mock_create_session.assert_called_once()
        assert session is mock_session
        assert ctx_mgr._session is mock_session

    @patch("zenoh_msgs.session._create_zenoh_session")
    def test_exit_closes_session(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        ctx_mgr = _ZenohSessionContextManager()
        ctx_mgr.__enter__()
        ctx_mgr.__exit__(None, None, None)

        mock_session.close.assert_called_once()

    @patch("zenoh_msgs.session._create_zenoh_session")
    def test_exit_handles_uninitialized_session(self, mock_create_session):
        ctx_mgr = _ZenohSessionContextManager()
        # Should not raise when session is None
        ctx_mgr.__exit__(None, None, None)

    @patch("zenoh_msgs.session._create_zenoh_session")
    def test_getattr_creates_session_lazily(self, mock_create_session):
        mock_session = MagicMock()
        mock_session.some_attr = "value"
        mock_create_session.return_value = mock_session

        ctx_mgr = _ZenohSessionContextManager()
        result = ctx_mgr.some_attr

        mock_create_session.assert_called_once()
        assert result == "value"
        assert ctx_mgr._session is mock_session

    @patch("zenoh_msgs.session._create_zenoh_session")
    def test_getattr_uses_cached_session(self, mock_create_session):
        mock_session = MagicMock()
        mock_session.attr1 = "value1"
        mock_session.attr2 = "value2"
        mock_create_session.return_value = mock_session

        ctx_mgr = _ZenohSessionContextManager()
        result1 = ctx_mgr.attr1
        result2 = ctx_mgr.attr2

        # Should only create session once
        mock_create_session.assert_called_once()
        assert result1 == "value1"
        assert result2 == "value2"

    @patch("zenoh_msgs.session._create_zenoh_session")
    def test_context_manager_with_statement(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        with open_zenoh_session() as session:
            assert session is mock_session

        mock_session.close.assert_called_once()


class TestCreateZenohSession:
    """Test the _create_zenoh_session function."""

    @pytest.fixture(autouse=True)
    def reset_hybrid_state(self):
        """Reset HybridZenohSession state between tests."""
        prev_use_sim = HybridZenohSession._use_sim
        HybridZenohSession._use_sim = False
        yield
        HybridZenohSession._use_sim = prev_use_sim

    def test_returns_hybrid_when_use_sim_true(self):
        HybridZenohSession.set_use_sim(True)
        session = _create_zenoh_session()
        assert isinstance(session, HybridZenohSession)

    @patch.object(zenoh, "open")
    def test_returns_zenoh_session_without_fallback(self, mock_zenoh_open):
        mock_session = MagicMock()
        mock_zenoh_open.return_value = mock_session

        HybridZenohSession.set_use_sim(False)
        session = _create_zenoh_session()

        mock_zenoh_open.assert_called_once()
        assert session is mock_session

    @patch.object(zenoh, "open")
    def test_fallback_to_network_discovery(self, mock_zenoh_open):
        mock_session_fallback = MagicMock()
        mock_zenoh_open.side_effect = [
            Exception("Local connection failed"),
            mock_session_fallback,
        ]

        HybridZenohSession.set_use_sim(False)
        session = _create_zenoh_session()

        assert mock_zenoh_open.call_count == 2
        assert session is mock_session_fallback

    @patch.object(zenoh, "open")
    def test_raises_when_all_attempts_fail(self, mock_zenoh_open):
        mock_zenoh_open.side_effect = [
            Exception("Local failed"),
            Exception("Fallback failed"),
        ]

        HybridZenohSession.set_use_sim(False)
        with pytest.raises(Exception, match="Failed to open Zenoh session"):
            _create_zenoh_session()

        assert mock_zenoh_open.call_count == 2


class TestHybridZenohSession:
    """Tests for the HybridZenohSession that routes between cloud + local."""

    @pytest.fixture(autouse=True)
    def reset_class_state(self):
        """Reset class-level config between tests."""
        prev_key = HybridZenohSession._api_key
        prev_use_sim = HybridZenohSession._use_sim
        HybridZenohSession._api_key = None
        HybridZenohSession._use_sim = False
        yield
        HybridZenohSession._api_key = prev_key
        HybridZenohSession._use_sim = prev_use_sim

    def test_set_api_key_class_level(self):
        HybridZenohSession.set_api_key("k")
        s = HybridZenohSession()
        assert s._api_key == "k"

    def test_set_use_sim_class_level(self):
        HybridZenohSession.set_use_sim(True)
        s = HybridZenohSession()
        assert s.use_sim is True

    def test_init_lazy_sessions_unset(self):
        s = HybridZenohSession()
        assert s._cloud_session is None
        assert s._zenoh_session is None

    @patch("zenoh_msgs.session._open_cloud_session")
    def test_get_cloud_session_lazy(self, mock_open_cloud):
        cloud = MagicMock()
        mock_open_cloud.return_value = cloud
        s = HybridZenohSession()
        first = s._get_cloud_session()
        second = s._get_cloud_session()
        assert first is cloud
        assert second is cloud
        # only opened once
        mock_open_cloud.assert_called_once()

    @patch.object(zenoh, "open")
    def test_get_zenoh_session_no_discovery_first(self, mock_zenoh_open):
        sess = MagicMock()
        mock_zenoh_open.return_value = sess
        s = HybridZenohSession()
        out = s._get_zenoh_session()
        assert out is sess
        # second call returns cached
        assert s._get_zenoh_session() is sess
        assert mock_zenoh_open.call_count == 1

    @patch.object(zenoh, "open")
    def test_get_zenoh_session_falls_back(self, mock_zenoh_open):
        sess = MagicMock()
        mock_zenoh_open.side_effect = [Exception("local"), sess]
        s = HybridZenohSession()
        out = s._get_zenoh_session()
        assert out is sess
        assert mock_zenoh_open.call_count == 2

    @patch.object(zenoh, "open")
    def test_get_zenoh_session_all_fail(self, mock_zenoh_open):
        mock_zenoh_open.side_effect = [Exception("a"), Exception("b")]
        s = HybridZenohSession()
        with pytest.raises(Exception, match="Failed to open Zenoh session"):
            s._get_zenoh_session()

    @patch("zenoh_msgs.session._open_cloud_session")
    def test_declare_subscriber_routes_cloud_for_known_topic(self, mock_open_cloud):
        cloud = MagicMock()
        mock_open_cloud.return_value = cloud
        s = HybridZenohSession()
        s.declare_subscriber("odom", lambda _x: None)
        cloud.declare_subscriber.assert_called_once()
        args, _ = cloud.declare_subscriber.call_args
        assert args[0] == "odom"

    @patch.object(zenoh, "open")
    def test_declare_subscriber_routes_local_for_unknown_topic(self, mock_zenoh_open):
        sess = MagicMock()
        mock_zenoh_open.return_value = sess
        s = HybridZenohSession()
        s.declare_subscriber("not_a_cloud_topic", lambda _x: None)
        sess.declare_subscriber.assert_called_once()

    @patch("zenoh_msgs.session._open_cloud_session")
    def test_declare_publisher_routes_cloud(self, mock_open_cloud):
        cloud = MagicMock()
        mock_open_cloud.return_value = cloud
        s = HybridZenohSession()
        s.declare_publisher("odom")
        cloud.declare_publisher.assert_called_once_with("odom")

    @patch.object(zenoh, "open")
    def test_declare_publisher_routes_local_for_unknown(self, mock_zenoh_open):
        sess = MagicMock()
        mock_zenoh_open.return_value = sess
        s = HybridZenohSession()
        s.declare_publisher("custom_topic")
        sess.declare_publisher.assert_called_once_with("custom_topic")

    @patch.object(zenoh, "open")
    def test_get_always_routes_local(self, mock_zenoh_open):
        sess = MagicMock()
        mock_zenoh_open.return_value = sess
        s = HybridZenohSession()
        s.get("custom_topic")
        s.get("odom")  # cloud topic still routed to local with warning
        assert sess.get.call_count == 2

    @patch("zenoh_msgs.session._open_cloud_session")
    def test_put_routes_cloud_for_known(self, mock_open_cloud):
        cloud = MagicMock()
        mock_open_cloud.return_value = cloud
        s = HybridZenohSession()
        s.put("odom", b"\x00")
        cloud.put.assert_called_once()

    @patch.object(zenoh, "open")
    def test_put_routes_local_for_unknown(self, mock_zenoh_open):
        sess = MagicMock()
        mock_zenoh_open.return_value = sess
        s = HybridZenohSession()
        s.put("custom_topic", b"\x00")
        sess.put.assert_called_once()

    @patch.object(zenoh, "open")
    @patch("zenoh_msgs.session._open_cloud_session")
    def test_close_closes_both(self, mock_open_cloud, mock_zenoh_open):
        cloud = MagicMock()
        mock_open_cloud.return_value = cloud
        sess = MagicMock()
        mock_zenoh_open.return_value = sess
        s = HybridZenohSession()
        # Force both to materialize
        s._get_cloud_session()
        s._get_zenoh_session()
        s.close()
        cloud.close.assert_called_once()
        sess.close.assert_called_once()

    def test_close_skips_when_uninitialized(self):
        s = HybridZenohSession()
        # Both lazy sessions are None — should be a no-op, not raise
        s.close()


class TestLoaders:
    @pytest.fixture(autouse=True)
    def isolate(self):
        prev_key = HybridZenohSession._api_key
        prev_sim = HybridZenohSession._use_sim
        HybridZenohSession._api_key = None
        HybridZenohSession._use_sim = False
        yield
        HybridZenohSession._api_key = prev_key
        HybridZenohSession._use_sim = prev_sim

    def test_load_api_key_with_value(self):
        load_api_key("xyz")
        assert HybridZenohSession._api_key == "xyz"

    def test_load_api_key_skips_empty(self):
        load_api_key(None)
        assert HybridZenohSession._api_key is None
        load_api_key("")
        assert HybridZenohSession._api_key is None

    def test_load_use_sim(self):
        load_use_sim(True)
        assert HybridZenohSession._use_sim is True
        load_use_sim(False)
        assert HybridZenohSession._use_sim is False

    def test_load_session_config_sets_both(self):
        load_session_config("api", True)
        assert HybridZenohSession._api_key == "api"
        assert HybridZenohSession._use_sim is True
