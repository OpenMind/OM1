"""Tests for expose.websim_adapter — WebSim wrapper."""

import socket
from unittest.mock import MagicMock

import pytest

from expose.websim_adapter import WebSimAdapter


@pytest.fixture
def fake_websim():
    return MagicMock()


class TestToolCalls:
    def test_move_sends_move_action_to_websim(self, fake_websim):
        adapter = WebSimAdapter(fake_websim)
        adapter.move("forward")
        args, _ = fake_websim.sim.call_args
        actions = args[0]
        assert len(actions) == 1
        assert actions[0].type == "move"
        assert actions[0].value == "forward"

    def test_speak_sends_speak_action(self, fake_websim):
        adapter = WebSimAdapter(fake_websim)
        adapter.speak("hello")
        actions = fake_websim.sim.call_args[0][0]
        assert actions[0].type == "speak"
        assert actions[0].value == "hello"

    def test_face_sends_emotion_action(self, fake_websim):
        adapter = WebSimAdapter(fake_websim)
        adapter.face("joy")
        actions = fake_websim.sim.call_args[0][0]
        assert actions[0].type == "emotion"
        assert actions[0].value == "joy"


class TestEnsurePortFree:
    def test_passes_when_port_unused(self):
        # Port 0 tells OS to pick a free port; close it and check a likely-free port
        WebSimAdapter.ensure_port_free("127.0.0.1", 59999)

    def test_raises_when_port_occupied(self):
        # Hold a port, then assert the check raises
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        held_port = s.getsockname()[1]
        try:
            with pytest.raises(RuntimeError, match="already in use"):
                WebSimAdapter.ensure_port_free("127.0.0.1", held_port)
        finally:
            s.close()
