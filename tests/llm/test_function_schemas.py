from dataclasses import dataclass
from typing import Optional

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface
from llm.function_schemas import generate_function_schema_from_action


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


@pytest.fixture
def action_config():
    return ActionConfig()


@pytest.fixture
def test_connector(action_config):
    return SampleConnector(action_config)


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
