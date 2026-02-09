"""Tests for the Privacy Guard WebSocket connector."""

from unittest.mock import AsyncMock, patch

import pytest

from actions.privacy_guard.interface import PrivacyAction, PrivacyGuardInput


class TestPrivacyGuardWebSocketConnector:
    """Tests for PrivacyGuardWebSocketConnector."""

    @pytest.fixture
    def connector(self):
        from actions.privacy_guard.connector.websocket import (
            PrivacyGuardWebSocketConnector,
            PrivacyGuardWSConfig,
        )

        config = PrivacyGuardWSConfig(
            look_away_angle=90,
            host="localhost",
            port=8000,
        )
        return PrivacyGuardWebSocketConnector(config)

    @pytest.mark.asyncio
    async def test_look_away_payload(self, connector):
        """look_away should send a payload with max effort_score."""
        with patch.object(
            connector, "_send_payload", new_callable=AsyncMock
        ) as mock_send:
            inp = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
            await connector.connect(inp)

            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert payload["action"] == "look_away"
            assert payload["effort_score"] == "max"
            assert payload["params"]["angle"] == 90
            assert connector._is_looking_away is True

    @pytest.mark.asyncio
    async def test_resume_payload(self, connector):
        """resume should send a payload with low effort_score."""
        connector._is_looking_away = True
        with patch.object(
            connector, "_send_payload", new_callable=AsyncMock
        ) as mock_send:
            inp = PrivacyGuardInput(action=PrivacyAction.RESUME)
            await connector.connect(inp)

            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert payload["action"] == "resume"
            assert payload["effort_score"] == "low"
            assert connector._is_looking_away is False

    @pytest.mark.asyncio
    async def test_stop_stream_payload(self, connector):
        """stop_stream should send with stream_killed flag."""
        with patch.object(
            connector, "_send_payload", new_callable=AsyncMock
        ) as mock_send:
            inp = PrivacyGuardInput(action=PrivacyAction.STOP_STREAM)
            await connector.connect(inp)

            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert payload["action"] == "stop_stream"
            assert payload["effort_score"] == "max"
            assert payload["params"]["stream_killed"] is True

    @pytest.mark.asyncio
    async def test_idle_payload(self, connector):
        """idle should send with low effort_score."""
        with patch.object(
            connector, "_send_payload", new_callable=AsyncMock
        ) as mock_send:
            inp = PrivacyGuardInput(action=PrivacyAction.IDLE)
            await connector.connect(inp)

            mock_send.assert_called_once()
            payload = mock_send.call_args[0][0]
            assert payload["action"] == "idle"
            assert payload["effort_score"] == "low"

    @pytest.mark.asyncio
    async def test_all_payloads_have_type(self, connector):
        """Every payload should have type = 'privacy_guard'."""
        for action in PrivacyAction:
            connector._is_looking_away = True  # ensure resume path is taken
            with patch.object(
                connector, "_send_payload", new_callable=AsyncMock
            ) as mock_send:
                inp = PrivacyGuardInput(action=action)
                await connector.connect(inp)
                payload = mock_send.call_args[0][0]
                assert payload["type"] == "privacy_guard"
