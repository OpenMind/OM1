from unittest.mock import MagicMock, call, patch

import pytest
import zenoh

from zenoh_msgs.session import create_zenoh_config, open_zenoh_session


class TestCreateZenohConfig:
    def test_create_config_with_network_discovery_enabled(self):
        config = create_zenoh_config()
        assert isinstance(config, zenoh.Config)

    def test_create_config_with_network_discovery_disabled(self):
        config = create_zenoh_config(network_discovery=False)
        assert isinstance(config, zenoh.Config)


class TestOpenZenohSession:
    @patch("zenoh_msgs.session.zenoh.open")
    def test_open_session_success_without_fallback(self, mock_zenoh_open):
        mock_session = MagicMock()
        mock_zenoh_open.return_value = mock_session

        with patch("zenoh_msgs.session.create_zenoh_config") as mock_create_config:
            local_config_mock = MagicMock()
            mock_create_config.side_effect = [local_config_mock, MagicMock()]

            session = open_zenoh_session()

            mock_create_config.assert_called_once_with(network_discovery=False)
            mock_zenoh_open.assert_called_once_with(local_config_mock)
            assert session is mock_session

    @patch("zenoh_msgs.session.zenoh.open")
    def test_open_session_fallback_success(self, mock_zenoh_open):
        mock_session_fallback = MagicMock()
        mock_zenoh_open.side_effect = [
            Exception("Local connection failed"),
            mock_session_fallback,
        ]

        with patch("zenoh_msgs.session.create_zenoh_config") as mock_create_config:
            local_config_mock = MagicMock()
            fallback_config_mock = MagicMock()
            mock_create_config.side_effect = [local_config_mock, fallback_config_mock]

            session = open_zenoh_session()

            expected_calls_to_create_config = [call(network_discovery=False), call()]
            mock_create_config.assert_has_calls(expected_calls_to_create_config)

            expected_calls_to_zenoh_open = [
                call(local_config_mock),
                call(fallback_config_mock),
            ]
            mock_zenoh_open.assert_has_calls(expected_calls_to_zenoh_open)
            assert session is mock_session_fallback

    @patch("zenoh_msgs.session.zenoh.open")
    def test_open_session_all_attempts_fail(self, mock_zenoh_open):
        mock_zenoh_open.side_effect = [
            Exception("Local failed"),
            Exception("Fallback failed"),
        ]

        with patch("zenoh_msgs.session.create_zenoh_config") as mock_create_config:
            local_config_mock = MagicMock()
            fallback_config_mock = MagicMock()
            mock_create_config.side_effect = [local_config_mock, fallback_config_mock]

            with pytest.raises(Exception, match="Failed to open Zenoh session"):
                open_zenoh_session()

            expected_calls_to_create_config = [call(network_discovery=False), call()]
            mock_create_config.assert_has_calls(expected_calls_to_create_config)

            expected_calls_to_zenoh_open = [
                call(local_config_mock),
                call(fallback_config_mock),
            ]
            mock_zenoh_open.assert_has_calls(expected_calls_to_zenoh_open)
