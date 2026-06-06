"""MCP tool definitions exposed by the OM1 server."""

from __future__ import annotations

from enum import Enum

import mcp.types as types


class MoveAction(str, Enum):
    """Supported values for the ``om1_move`` tool's ``action`` parameter."""

    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SPIN = "spin"
    SIT = "sit"
    STAND = "stand"
    IDLE = "idle"


class Emotion(str, Enum):
    """Supported values for the ``om1_face`` tool's ``emotion`` parameter."""

    JOY = "joy"
    SMILE = "smile"
    PONDER = "ponder"
    ALERT = "alert"
    SAD = "sad"


def build_tool_definitions() -> list[types.Tool]:
    """Return the list of MCP Tool schemas exposed by the OM1 server."""
    return [
        types.Tool(
            name="om1_move",
            description="Move the OM1 agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [a.value for a in MoveAction],
                    }
                },
                "required": ["action"],
            },
        ),
        types.Tool(
            name="om1_speak",
            description="Make the OM1 agent speak text aloud.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 500,
                    }
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="om1_face",
            description="Change the OM1 agent's facial emotion.",
            inputSchema={
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": [e.value for e in Emotion],
                    }
                },
                "required": ["emotion"],
            },
        ),
    ]
