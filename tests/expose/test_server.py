"""Tests for expose.server — tool dispatch and MCP wiring."""

from unittest.mock import MagicMock

import pytest

from expose.server import handle_tool_call


@pytest.fixture
def adapter():
    return MagicMock()


class TestMove:
    @pytest.mark.asyncio
    async def test_valid_move_calls_adapter(self, adapter):
        result = await handle_tool_call("om1_move", {"action": "forward"}, adapter)
        adapter.move.assert_called_once_with("forward")
        assert len(result) == 1
        assert "forward" in result[0].text

    @pytest.mark.asyncio
    async def test_invalid_move_returns_error_not_raise(self, adapter):
        result = await handle_tool_call("om1_move", {"action": "moonwalk"}, adapter)
        adapter.move.assert_not_called()
        assert result[0].text.lower().startswith("error")

    @pytest.mark.asyncio
    async def test_missing_action_arg_returns_error(self, adapter):
        result = await handle_tool_call("om1_move", {}, adapter)
        adapter.move.assert_not_called()
        assert "error" in result[0].text.lower()


class TestSpeak:
    @pytest.mark.asyncio
    async def test_valid_speak_calls_adapter(self, adapter):
        result = await handle_tool_call("om1_speak", {"text": "hi"}, adapter)
        adapter.speak.assert_called_once_with("hi")
        assert "hi" in result[0].text

    @pytest.mark.asyncio
    async def test_empty_text_returns_error(self, adapter):
        result = await handle_tool_call("om1_speak", {"text": "   "}, adapter)
        adapter.speak.assert_not_called()
        assert "error" in result[0].text.lower()


class TestFace:
    @pytest.mark.asyncio
    async def test_valid_emotion(self, adapter):
        result = await handle_tool_call("om1_face", {"emotion": "joy"}, adapter)
        adapter.face.assert_called_once_with("joy")
        assert "joy" in result[0].text

    @pytest.mark.asyncio
    async def test_invalid_emotion_returns_error(self, adapter):
        result = await handle_tool_call("om1_face", {"emotion": "angry"}, adapter)
        adapter.face.assert_not_called()
        assert "error" in result[0].text.lower()


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, adapter):
        result = await handle_tool_call("om1_teleport", {}, adapter)
        assert "error" in result[0].text.lower()
        assert "om1_teleport" in result[0].text
