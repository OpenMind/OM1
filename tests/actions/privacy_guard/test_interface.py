"""Tests for the Privacy Guard action interface."""

import pytest

from actions.privacy_guard.interface import (
    PrivacyAction,
    PrivacyGuard,
    PrivacyGuardInput,
)


class TestPrivacyAction:
    """Tests for the PrivacyAction enum."""

    def test_enum_values(self):
        assert PrivacyAction.LOOK_AWAY == "look_away"
        assert PrivacyAction.RESUME == "resume"
        assert PrivacyAction.STOP_STREAM == "stop_stream"
        assert PrivacyAction.IDLE == "idle"

    def test_enum_count(self):
        assert len(PrivacyAction) == 4

    def test_enum_from_string(self):
        assert PrivacyAction("look_away") == PrivacyAction.LOOK_AWAY
        assert PrivacyAction("resume") == PrivacyAction.RESUME
        assert PrivacyAction("stop_stream") == PrivacyAction.STOP_STREAM
        assert PrivacyAction("idle") == PrivacyAction.IDLE

    def test_enum_invalid_value(self):
        with pytest.raises(ValueError):
            PrivacyAction("invalid_action")

    def test_enum_is_str(self):
        """PrivacyAction inherits from str, so values should be usable as strings."""
        action = PrivacyAction.LOOK_AWAY
        assert isinstance(action, str)
        assert action == "look_away"


class TestPrivacyGuardInput:
    """Tests for the PrivacyGuardInput dataclass."""

    def test_create_with_enum(self):
        inp = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        assert inp.action == PrivacyAction.LOOK_AWAY

    def test_create_all_actions(self):
        for action in PrivacyAction:
            inp = PrivacyGuardInput(action=action)
            assert inp.action == action


class TestPrivacyGuardInterface:
    """Tests for the PrivacyGuard Interface dataclass."""

    def test_create_interface(self):
        inp = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        out = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        guard = PrivacyGuard(input=inp, output=out)
        assert guard.input.action == PrivacyAction.LOOK_AWAY
        assert guard.output.action == PrivacyAction.LOOK_AWAY

    def test_interface_docstring_exists(self):
        """The docstring is used as the LLM prompt via describe_action()."""
        assert PrivacyGuard.__doc__ is not None
        assert "look_away" in PrivacyGuard.__doc__
        assert "resume" in PrivacyGuard.__doc__
        assert "stop_stream" in PrivacyGuard.__doc__
        assert "idle" in PrivacyGuard.__doc__
