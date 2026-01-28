from dataclasses import dataclass
from typing import Optional

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface


@dataclass
class SampleInput:
    value: str


@dataclass
class SampleOutput:
    result: str


# Test implementation of Interface
@dataclass
class SampleInterface(Interface[SampleInput, SampleOutput]):
    input: SampleInput
    output: SampleOutput


# Test implementation of ActionConnector
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


@pytest.mark.asyncio
async def test_connector_connect():
    config = ActionConfig()
    connector = SampleConnector(config)
    test_output = SampleOutput(result="test_result")

    await connector.connect(test_output)

    assert connector.last_output == test_output


@pytest.mark.asyncio
async def test_full_action_flow(agent_action):
    test_input = SampleInput(value="test_data")

    # Connect the output
    await agent_action.connector.connect(test_input)
    assert isinstance(agent_action.connector, SampleConnector)
    assert agent_action.connector.last_output == test_input


def test_action_config():
    config = ActionConfig()

    assert config is not None
    assert isinstance(config, ActionConfig)


def test_agent_action_structure(agent_action):
    assert agent_action.name == "test_action"
    assert agent_action.interface == SampleInterface
    assert isinstance(agent_action.connector, SampleConnector)
