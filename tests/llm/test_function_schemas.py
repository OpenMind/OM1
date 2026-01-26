import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface
from llm.function_schemas import (
    convert_function_calls_to_actions,
    generate_function_schema_from_action,
    generate_function_schemas_from_actions,
)


@dataclass
class SampleInput:
    value: str


@dataclass
class SampleOutput:
    result: str


@dataclass
class SampleInterface(Interface[SampleInput, SampleOutput]):
    input: SampleInput
    output: SampleOutput


class SampleConnector(ActionConnector[ActionConfig, SampleOutput]):
    def __init__(self, config: ActionConfig):
        super().__init__(config)
        self.last_output: Optional[SampleOutput] = None

    async def connect(self, output_interface: SampleOutput) -> None:
        self.last_output = output_interface


class SampleStrEnum(str, Enum):
    WALK = "walk"
    RUN = "run"
    STOP = "stop"


class SampleIntEnum(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class IntFieldInput:
    timeout_sec: int


@dataclass
class IntFieldInterface(Interface[IntFieldInput, IntFieldInput]):
    """Action with an integer parameter."""

    input: IntFieldInput
    output: IntFieldInput


@dataclass
class FloatFieldInput:
    speed: float


@dataclass
class FloatFieldInterface(Interface[FloatFieldInput, FloatFieldInput]):
    """Action with a float parameter."""

    input: FloatFieldInput
    output: FloatFieldInput


@dataclass
class BoolFieldInput:
    enabled: bool


@dataclass
class BoolFieldInterface(Interface[BoolFieldInput, BoolFieldInput]):
    """Action with a boolean parameter."""

    input: BoolFieldInput
    output: BoolFieldInput


@dataclass
class StrEnumFieldInput:
    action: SampleStrEnum


@dataclass
class StrEnumFieldInterface(Interface[StrEnumFieldInput, StrEnumFieldInput]):
    """Action with a string enum parameter."""

    input: StrEnumFieldInput
    output: StrEnumFieldInput


@dataclass
class IntEnumFieldInput:
    priority: SampleIntEnum


@dataclass
class IntEnumFieldInterface(Interface[IntEnumFieldInput, IntEnumFieldInput]):
    """Action with an integer enum parameter."""

    input: IntEnumFieldInput
    output: IntEnumFieldInput


@dataclass
class MixedFieldInput:
    action: str
    count: int
    speed: float
    enabled: bool


@dataclass
class MixedFieldInterface(Interface[MixedFieldInput, MixedFieldInput]):
    """Action with mixed parameter types."""

    input: MixedFieldInput
    output: MixedFieldInput


@pytest.fixture
def action_config():
    return ActionConfig()


@pytest.fixture
def test_connector(action_config):
    return SampleConnector(action_config)


def _make_agent_action(test_connector, interface, label="test_label"):
    return AgentAction(
        name="test_action",
        llm_label=label,
        interface=interface,
        connector=test_connector,
        exclude_from_prompt=False,
    )


@pytest.fixture
def agent_action(test_connector):
    return AgentAction(
        name="test_action",
        llm_label="test_llm_label",
        interface=SampleInterface,
        connector=test_connector,
        exclude_from_prompt=True,
    )


def test_generate_function_schema_from_action(agent_action):
    schema = generate_function_schema_from_action(agent_action)

    assert "function" in schema
    assert schema["type"] == "function"

    fn = schema["function"]
    assert "description" in fn
    assert "parameters" in fn
    assert fn["name"] == "test_llm_label"

    params = fn["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "value" in params["properties"]

    value_prop = params["properties"]["value"]
    assert value_prop["type"] == "string"
    assert "description" in value_prop

    assert params["required"] == ["value"]
    assert fn["description"].startswith("SampleInterface(")


def test_convert_single_action_parameter():
    """Test conversion with single 'action' parameter."""
    function_calls = [
        {
            "function": {
                "name": "move",
                "arguments": json.dumps({"action": "forward"}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "move"
    assert actions[0].value == "forward"


def test_convert_multiple_parameters_json_dumps():
    """Test conversion with multiple parameters - should JSON dump them."""
    function_calls = [
        {
            "function": {
                "name": "greeting_conversation",
                "arguments": json.dumps(
                    {
                        "response": "Hello there!",
                        "conversation_state": "conversing",
                        "confidence": 0.95,
                        "speech_clarity": 0.8,
                    }
                ),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "greeting_conversation"
    value = json.loads(actions[0].value)
    assert value["response"] == "Hello there!"
    assert value["conversation_state"] == "conversing"
    assert value["confidence"] == 0.95
    assert value["speech_clarity"] == 0.8


def test_convert_single_text_parameter():
    """Test conversion with single 'text' parameter (common fallback)."""
    function_calls = [
        {
            "function": {
                "name": "speak",
                "arguments": json.dumps({"text": "Hello world"}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "speak"
    assert actions[0].value == "Hello world"


def test_convert_single_message_parameter():
    """Test conversion with single 'message' parameter (common fallback)."""
    function_calls = [
        {
            "function": {
                "name": "notify",
                "arguments": json.dumps({"message": "Task complete"}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "notify"
    assert actions[0].value == "Task complete"


def test_convert_single_value_parameter():
    """Test conversion with single 'value' parameter (common fallback)."""
    function_calls = [
        {
            "function": {
                "name": "set_speed",
                "arguments": json.dumps({"value": "fast"}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "set_speed"
    assert actions[0].value == "fast"


def test_convert_single_command_parameter():
    """Test conversion with single 'command' parameter (common fallback)."""
    function_calls = [
        {
            "function": {
                "name": "execute",
                "arguments": json.dumps({"command": "shutdown"}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "execute"
    assert actions[0].value == "shutdown"


def test_convert_single_arbitrary_parameter():
    """Test conversion with single arbitrary parameter (not in common list)."""
    function_calls = [
        {
            "function": {
                "name": "move",
                "arguments": json.dumps({"direction": "north"}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "move"
    assert actions[0].value == "north"


def test_convert_empty_arguments():
    """Test conversion with empty arguments."""
    function_calls = [{"function": {"name": "stop", "arguments": json.dumps({})}}]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "stop"
    assert actions[0].value == ""


def test_convert_arguments_already_dict():
    """Test conversion when arguments is already a dict (not a JSON string)."""
    function_calls = [
        {"function": {"name": "move", "arguments": {"action": "forward"}}}
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "move"
    assert actions[0].value == "forward"


def test_convert_invalid_json_arguments():
    """Test handling of invalid JSON in arguments - should skip."""
    function_calls = [
        {"function": {"name": "move", "arguments": "{invalid json}"}},
        {"function": {"name": "stop", "arguments": json.dumps({"action": "now"})}},
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "stop"
    assert actions[0].value == "now"


def test_convert_multiple_function_calls():
    """Test conversion of multiple function calls."""
    function_calls = [
        {
            "function": {
                "name": "move",
                "arguments": json.dumps({"action": "forward"}),
            }
        },
        {
            "function": {
                "name": "speak",
                "arguments": json.dumps({"text": "Moving forward"}),
            }
        },
        {
            "function": {
                "name": "rotate",
                "arguments": json.dumps({"angle": 90, "speed": 1.5}),
            }
        },
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 3
    assert actions[0].type == "move"
    assert actions[0].value == "forward"
    assert actions[1].type == "speak"
    assert actions[1].value == "Moving forward"
    assert actions[2].type == "rotate"

    rotate_value = json.loads(actions[2].value)
    assert rotate_value["angle"] == 90
    assert rotate_value["speed"] == 1.5


def test_convert_missing_function_name():
    """Test handling when function name is missing - should skip due to validation error."""
    function_calls = [{"function": {"arguments": json.dumps({"action": "test"})}}]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 0


def test_convert_missing_arguments():
    """Test handling when arguments are missing - should use empty dict."""
    function_calls = [{"function": {"name": "stop"}}]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "stop"
    assert actions[0].value == ""


def test_convert_numeric_values():
    """Test conversion with numeric values."""
    function_calls = [
        {
            "function": {
                "name": "set_speed",
                "arguments": json.dumps({"speed": 42}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "set_speed"
    assert actions[0].value == "42"


def test_convert_boolean_values():
    """Test conversion with boolean values."""
    function_calls = [
        {
            "function": {
                "name": "set_enabled",
                "arguments": json.dumps({"enabled": True}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "set_enabled"
    assert actions[0].value == "True"


def test_convert_null_values():
    """Test conversion with null/None values - should skip due to validation error."""
    function_calls = [
        {"function": {"name": "reset", "arguments": json.dumps({"value": None})}}
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 0


def test_convert_priority_of_action_parameter():
    """Test that 'action' parameter takes priority when it's the only parameter."""
    function_calls = [
        {
            "function": {
                "name": "move",
                "arguments": json.dumps({"action": "forward"}),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].value == "forward"


def test_convert_fallback_parameter_priority():
    """Test fallback parameter priority: text > message > value > command."""
    function_calls = [
        {
            "function": {
                "name": "cmd1",
                "arguments": json.dumps({"text": "hello"}),
            }
        }
    ]
    actions = convert_function_calls_to_actions(function_calls)
    assert actions[0].value == "hello"


def test_convert_exception_handling():
    """Test that exceptions in conversion are caught and processing continues."""
    function_calls = [
        {"function": {"name": "move", "arguments": json.dumps({"action": "ok"})}},
        {"invalid": "structure"},
        {"function": {"name": "stop", "arguments": json.dumps({"action": "now"})}},
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 2
    assert actions[0].type == "move"
    assert actions[1].type == "stop"


def test_convert_empty_function_calls_list():
    """Test conversion with empty function calls list."""
    function_calls = []

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 0
    assert actions == []


def test_convert_complex_nested_json_in_multiple_params():
    """Test complex nested structures with multiple parameters."""
    function_calls = [
        {
            "function": {
                "name": "configure",
                "arguments": json.dumps(
                    {
                        "mode": "auto",
                        "settings": {"speed": 1.5, "direction": "north"},
                        "enabled": True,
                    }
                ),
            }
        }
    ]

    actions = convert_function_calls_to_actions(function_calls)

    assert len(actions) == 1
    assert actions[0].type == "configure"

    value = json.loads(actions[0].value)
    assert value["mode"] == "auto"
    assert value["settings"]["speed"] == 1.5
    assert value["settings"]["direction"] == "north"
    assert value["enabled"] is True


def test_int_field_generates_integer_type(test_connector):
    action = _make_agent_action(test_connector, IntFieldInterface, "int_action")
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert (
        props["timeout_sec"]["type"] == "integer"
    ), f"int field should produce 'integer' type, got '{props['timeout_sec']['type']}'"


def test_float_field_generates_number_type(test_connector):
    action = _make_agent_action(test_connector, FloatFieldInterface, "float_action")
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert (
        props["speed"]["type"] == "number"
    ), f"float field should produce 'number' type, got '{props['speed']['type']}'"


def test_bool_field_generates_boolean_type(test_connector):
    action = _make_agent_action(test_connector, BoolFieldInterface, "bool_action")
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert (
        props["enabled"]["type"] == "boolean"
    ), f"bool field should produce 'boolean' type, got '{props['enabled']['type']}'"


def test_str_enum_field_generates_enum(test_connector):
    action = _make_agent_action(
        test_connector, StrEnumFieldInterface, "str_enum_action"
    )
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert props["action"]["type"] == "string"
    assert "enum" in props["action"]
    assert set(props["action"]["enum"]) == {"walk", "run", "stop"}


def test_int_enum_field_does_not_crash(test_connector):
    action = _make_agent_action(
        test_connector, IntEnumFieldInterface, "int_enum_action"
    )
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert "priority" in props
    assert "enum" in props["priority"]
    assert set(props["priority"]["enum"]) == {1, 2, 3}


def test_mixed_field_types(test_connector):
    action = _make_agent_action(test_connector, MixedFieldInterface, "mixed_action")
    schema = generate_function_schema_from_action(action)

    props = schema["function"]["parameters"]["properties"]
    assert props["action"]["type"] == "string"
    assert (
        props["count"]["type"] == "integer"
    ), f"int field should produce 'integer', got '{props['count']['type']}'"
    assert (
        props["speed"]["type"] == "number"
    ), f"float field should produce 'number', got '{props['speed']['type']}'"
    assert (
        props["enabled"]["type"] == "boolean"
    ), f"bool field should produce 'boolean', got '{props['enabled']['type']}'"

    required = schema["function"]["parameters"]["required"]
    assert set(required) == {"action", "count", "speed", "enabled"}


def test_schemas_excludes_excluded_actions(test_connector):
    action_included = _make_agent_action(
        test_connector, SampleInterface, "included_action"
    )
    action_excluded = AgentAction(
        name="excluded",
        llm_label="excluded_action",
        interface=SampleInterface,
        connector=test_connector,
        exclude_from_prompt=True,
    )
    schemas = generate_function_schemas_from_actions([action_included, action_excluded])
    names = [s["function"]["name"] for s in schemas]
    assert "included_action" in names
    assert "excluded_action" not in names
