"""Tests for expose.tools — MCP tool schema definitions."""

from expose.tools import Emotion, MoveAction, build_tool_definitions


def _tool(name: str):
    return next(t for t in build_tool_definitions() if t.name == name)


class TestBuildToolDefinitions:
    def test_returns_three_tools_with_expected_names(self):
        tools = build_tool_definitions()
        names = {t.name for t in tools}
        assert names == {"om1_move", "om1_speak", "om1_face"}


class TestMoveSchema:
    def test_move_action_param_is_enum_matching_move_action_enum(self):
        schema = _tool("om1_move").inputSchema
        action = schema["properties"]["action"]
        assert set(action["enum"]) == {a.value for a in MoveAction}
        assert schema["required"] == ["action"]

    def test_move_action_enum_contains_core_values(self):
        values = {a.value for a in MoveAction}
        assert {"forward", "backward", "turn_left", "turn_right", "spin"} <= values


class TestFaceSchema:
    def test_face_emotion_param_is_enum_matching_emotion_enum(self):
        schema = _tool("om1_face").inputSchema
        emotion = schema["properties"]["emotion"]
        assert set(emotion["enum"]) == {e.value for e in Emotion}
        assert schema["required"] == ["emotion"]

    def test_emotion_enum_has_five_values(self):
        assert {e.value for e in Emotion} == {"joy", "smile", "ponder", "alert", "sad"}


class TestSpeakSchema:
    def test_speak_text_has_length_constraints(self):
        schema = _tool("om1_speak").inputSchema
        text = schema["properties"]["text"]
        assert text["type"] == "string"
        assert text["minLength"] == 1
        assert text["maxLength"] == 500
        assert schema["required"] == ["text"]
