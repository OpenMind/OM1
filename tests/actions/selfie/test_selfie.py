# tests/actions/selfie/test_selfie_connector.py
"""Unit tests for the Selfie action connector."""

import logging
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from actions.selfie.interface import SelfieInput

# Mock providers before importing the connector
sys.modules["providers.elevenlabs_tts_provider"] = MagicMock()
sys.modules["providers.io_provider"] = MagicMock()


class TestSelfieConfig:
    """Tests for SelfieConfig."""

    def test_config_default_values(self):
        """Test default configuration values."""
        from actions.selfie.connector.selfie import SelfieConfig

        config = SelfieConfig()
        assert config.face_http_base_url == "http://127.0.0.1:6793"
        assert config.face_recent_sec == 1.0
        assert config.poll_ms == 200
        assert config.timeout_sec == 15
        assert config.http_timeout_sec == 5.0


class TestSelfieConnector:
    """Tests for SelfieConnector."""

    @pytest.fixture
    def mock_providers(self):
        """Set up mock providers for the connector."""
        mock_tts = MagicMock()
        mock_io = MagicMock()
        return {"tts": mock_tts, "io": mock_io}

    def test_connector_initialization(self, mock_providers):
        """Test connector initialization."""
        with patch(
            "actions.selfie.connector.selfie.ElevenLabsTTSProvider",
            return_value=mock_providers["tts"],
        ):
            with patch(
                "actions.selfie.connector.selfie.IOProvider",
                return_value=mock_providers["io"],
            ):
                from actions.selfie.connector.selfie import (
                    SelfieConfig,
                    SelfieConnector,
                )

                config = SelfieConfig(face_http_base_url="http://test-face-service")
                connector = SelfieConnector(config)

                assert connector.base_url == "http://test-face-service"
                assert connector.elevenlabs_tts_provider == mock_providers["tts"]
                assert connector.io_provider == mock_providers["io"]

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_providers):
        """Test successful selfie enrollment."""
        with patch(
            "actions.selfie.connector.selfie.ElevenLabsTTSProvider",
            return_value=mock_providers["tts"],
        ):
            with patch(
                "actions.selfie.connector.selfie.IOProvider",
                return_value=mock_providers["io"],
            ):
                from actions.selfie.connector.selfie import (
                    SelfieConfig,
                    SelfieConnector,
                )

                connector = SelfieConnector(SelfieConfig())

                # Mock HTTP responses
                mock_resp_config = MagicMock()
                mock_resp_config.json.return_value = {"config": {"blur": True}}

                mock_resp_who = MagicMock()
                mock_resp_who.json.return_value = {"now": ["person1"], "unknown_now": 0}

                mock_resp_selfie = MagicMock()
                mock_resp_selfie.json.return_value = {"ok": True}

                with patch("requests.post") as mock_post:
                    mock_post.side_effect = [
                        mock_resp_config,
                        MagicMock(),
                        mock_resp_who,
                        mock_resp_selfie,
                        MagicMock(),
                    ]

                    selfie_input = SelfieInput(action="wendy")
                    await connector.connect(selfie_input)

                    # Verify final selfie call
                    # The second call is to set blur to False
                    # The third is _who_snapshot in _wait_single_face
                    # The fourth is /selfie
                    # The fifth is to restore blur

                    # Find the /selfie call
                    selfie_call = next(
                        c for c in mock_post.call_args_list if "/selfie" in c[0][0]
                    )
                    assert selfie_call[1]["json"]["id"] == "wendy"

                    # Verify status write
                    mock_providers["io"].add_input.assert_any_call(
                        "SelfieStatus", "ok id=wendy", pytest.approx(time.time(), abs=1)
                    )
                    # Verify TTS feedback
                    mock_providers["tts"].add_pending_message.assert_called()

    @pytest.mark.asyncio
    async def test_connect_bad_id_logs_error(self, mock_providers, caplog):
        """Test connect with empty ID fails and logs error."""
        with patch(
            "actions.selfie.connector.selfie.ElevenLabsTTSProvider",
            return_value=mock_providers["tts"],
        ):
            with patch(
                "actions.selfie.connector.selfie.IOProvider",
                return_value=mock_providers["io"],
            ):
                from actions.selfie.connector.selfie import (
                    SelfieConfig,
                    SelfieConnector,
                )

                connector = SelfieConnector(SelfieConfig())
                selfie_input = SelfieInput(action="")  # Empty ID

                with caplog.at_level(logging.ERROR):
                    await connector.connect(selfie_input)

                assert "requires a non-empty `id`" in caplog.text
                mock_providers["io"].add_input.assert_called_with(
                    "SelfieStatus",
                    "failed reason=bad_id",
                    pytest.approx(time.time(), abs=1),
                )

    @pytest.mark.asyncio
    async def test_connect_timeout_waiting_for_face(self, mock_providers, caplog):
        """Test handles timeout when no faces are detected."""
        with patch(
            "actions.selfie.connector.selfie.ElevenLabsTTSProvider",
            return_value=mock_providers["tts"],
        ):
            with patch(
                "actions.selfie.connector.selfie.IOProvider",
                return_value=mock_providers["io"],
            ):
                from actions.selfie.connector.selfie import (
                    SelfieConfig,
                    SelfieConnector,
                )

                # Shorten poll/timeout for test speed
                config = SelfieConfig(poll_ms=10, timeout_sec=1)
                connector = SelfieConnector(config)

                # Mock HTTP responses
                mock_resp_config = MagicMock()
                mock_resp_config.json.return_value = {"config": {"blur": True}}

                mock_resp_who_empty = MagicMock()
                mock_resp_who_empty.json.return_value = {"now": [], "unknown_now": 0}

                with patch("requests.post") as mock_post:
                    # _get_config, _set_blur, then many _who_snapshot, then _who_snapshot again for reason, then _set_blur
                    mock_post.return_value = mock_resp_who_empty
                    mock_post.side_effect = None  # Always return empty for who

                    # We need to handle the first few calls differently
                    def mock_responses(url, **kwargs):
                        m = MagicMock()
                        if "/config" in url:
                            m.json.return_value = {"config": {"blur": True}}
                        elif "/who" in url:
                            m.json.return_value = {"now": [], "unknown_now": 0}
                        return m

                    mock_post.side_effect = mock_responses

                    selfie_input = SelfieInput(action="tester", timeout_sec=0.1)
                    await connector.connect(selfie_input)

                    assert "Selfie gate: timeout" in caplog.text
                    mock_providers["io"].add_input.assert_any_call(
                        "SelfieStatus",
                        "failed reason=none faces=0",
                        pytest.approx(time.time(), abs=5),
                    )

    def test_connector_inherits_from_action_connector(self):
        """Test that SelfieConnector inherits from ActionConnector."""
        from actions.base import ActionConnector
        from actions.selfie.connector.selfie import SelfieConnector

        assert issubclass(SelfieConnector, ActionConnector)
