"""Unit tests for episodic_hook module."""

from unittest.mock import Mock, patch

import pytest

from hooks.episodic_hook import start_episodic_hook, stop_episodic_hook


@pytest.fixture
def mock_provider():
    """Mock EpisodicMemoryProvider."""
    with patch('hooks.episodic_hook.EpisodicMemoryProvider') as mock_cls:
        provider = Mock()
        mock_cls.return_value = provider
        yield provider


@pytest.fixture
def mock_tts():
    """Mock ElevenLabsTTSProvider."""
    with patch('hooks.episodic_hook.ElevenLabsTTSProvider') as mock_cls:
        tts = Mock()
        mock_cls.return_value = tts
        yield tts


class TestStartEpisodicHook:
    """Tests for start_episodic_hook function."""

    @pytest.mark.asyncio
    async def test_start_default_context(self, mock_provider):
        """Start hook with empty context should use default mode."""
        result = await start_episodic_hook({})
        assert result["status"] == "success"
        assert "default" in result["message"]
        mock_provider.load_mode.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_start_custom_mode(self, mock_provider):
        """Start hook should set current mode on provider."""
        result = await start_episodic_hook({"mode": "navigation"})
        assert result["status"] == "success"
        assert mock_provider._current_mode == "navigation"
        mock_provider.load_mode.assert_called_once_with("navigation")

    @pytest.mark.asyncio
    async def test_start_with_announce(self, mock_provider, mock_tts):
        """Start hook with announce=True should call TTS."""
        result = await start_episodic_hook({"mode": "slam", "announce": True})
        assert result["status"] == "success"
        mock_tts.add_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_without_announce(self, mock_provider, mock_tts):
        """Start hook with announce=False should not call TTS."""
        result = await start_episodic_hook({"mode": "slam", "announce": False})
        assert result["status"] == "success"
        mock_tts.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_tts_failure_is_non_fatal(self, mock_provider, mock_tts):
        """Start hook should succeed even if TTS announcement fails."""
        mock_tts.add_pending_message.side_effect = Exception("TTS unavailable")
        result = await start_episodic_hook({"mode": "slam", "announce": True})
        assert result["status"] == "success"


class TestStopEpisodicHook:
    """Tests for stop_episodic_hook function."""

    @pytest.mark.asyncio
    async def test_stop_default_context(self, mock_provider):
        """Stop hook with empty context should flush and succeed."""
        result = await stop_episodic_hook({})
        assert result["status"] == "success"
        mock_provider.save_to_disk.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_custom_mode(self, mock_provider):
        """Stop hook should include mode in result message."""
        result = await stop_episodic_hook({"mode": "navigation"})
        assert result["status"] == "success"
        assert "navigation" in result["message"]

    @pytest.mark.asyncio
    async def test_stop_with_announce(self, mock_provider, mock_tts):
        """Stop hook with announce=True should call TTS."""
        result = await stop_episodic_hook({"mode": "slam", "announce": True})
        assert result["status"] == "success"
        mock_tts.add_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_flush_failure_raises(self, mock_provider):
        """Stop hook should raise if save_to_disk fails."""
        mock_provider.save_to_disk.side_effect = Exception("disk full")
        with pytest.raises(Exception, match="disk full"):
            await stop_episodic_hook({})

    @pytest.mark.asyncio
    async def test_stop_tts_failure_is_non_fatal(self, mock_provider, mock_tts):
        """Stop hook should succeed even if TTS announcement fails."""
        mock_tts.add_pending_message.side_effect = Exception("TTS unavailable")
        result = await stop_episodic_hook({"mode": "slam", "announce": True})
        assert result["status"] == "success"
