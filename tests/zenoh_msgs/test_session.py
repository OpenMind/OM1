from unittest.mock import MagicMock, patch

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
    @patch.object(zenoh, "open")
    def test_open_session_success_without_fallback(self, mock_zenoh_open):
        mock_session = MagicMock()
        mock_zenoh_open.return_value = mock_session

        session = open_zenoh_session()

        mock_zenoh_open.assert_called_once()
        assert session is mock_session

    @patch.object(zenoh, "open")
    def test_open_session_fallback_success(self, mock_zenoh_open):
        mock_session_fallback = MagicMock()
        mock_zenoh_open.side_effect = [
            Exception("Local connection failed"),
            mock_session_fallback,
        ]

        session = open_zenoh_session()

        assert mock_zenoh_open.call_count == 2
        assert session is mock_session_fallback

    @patch.object(zenoh, "open")
    def test_open_session_all_attempts_fail(self, mock_zenoh_open):
        mock_zenoh_open.side_effect = [
            Exception("Local failed"),
            Exception("Fallback failed"),
        ]

        with pytest.raises(Exception, match="Failed to open Zenoh session"):
            open_zenoh_session()

        assert mock_zenoh_open.call_count == 2
        expected_calls_to_zenoh_open = mock_zenoh_open.call_args_list
        mock_zenoh_open.assert_has_calls(expected_calls_to_zenoh_open)
