import json
import logging
import os
from contextlib import contextmanager
from typing import Optional
from unittest.mock import mock_open, patch

import pytest

from actions.base import ActionConfig, ActionConnector, AgentAction, Interface
from inputs.base import Sensor, SensorConfig
from llm import LLM, LLMConfig
from llm.output_model import CortexOutputModel
from runtime.single_mode.config import RuntimeConfig, load_config
from simulators.base import Simulator, SimulatorConfig


@pytest.fixture
def mock_config_data():
    return {
        "version": "v1.0.1",
        "hertz": 10.0,
        "name": "test_config",
        "api_key": "global_test_api_key",
        "system_prompt_base": "system prompt base",
        "system_governance": "system governance",
        "system_prompt_examples": "system prompt examples",
        "agent_inputs": [{"type": "test_input"}],
        "cortex_llm": {"type": "test_llm", "config": {"model": "test-model"}},
        "simulators": [
            {"type": "test_simulator", "config": {"api_key": "sim_test_api_key"}}
        ],
        "agent_actions": [
            {
                "name": "test_action",
                "connector": "test_connector",
                "exclude_from_prompt": False,
                "config": {
                    "arg1": "val1",
                },
            }
        ],
    }


@pytest.fixture
def mock_dependencies():
    class MockInput(Sensor):
        def __init__(self, config=SensorConfig()):
            super().__init__(config)

    class MockInterface(Interface[str, str]):
        """Mock interface for testing"""

        pass

    class MockConnector(ActionConnector[ActionConfig, str]):
        """Mock connector for testing"""

        def __init__(self):
            super().__init__(ActionConfig())

        async def connect(self, output_interface: str) -> None:
            pass

    class MockAction(AgentAction):
        def __init__(self):
            super().__init__(
                name="mock_action",
                llm_label="mock_llm_label",
                interface=MockInterface,
                connector=MockConnector(),
                exclude_from_prompt=False,
            )

    class MockSimulator(Simulator):
        def __init__(self, config: SimulatorConfig = SimulatorConfig()):
            super().__init__(config)

    class MockLLM(LLM[CortexOutputModel]):
        def __init__(self, config: LLMConfig = LLMConfig()):
            super().__init__(config, available_actions=None)

    return {
        "input": MockInput,
        "action": MockAction,
        "simulator": MockSimulator,
        "llm": MockLLM,
    }


@pytest.fixture
def mock_empty_config_data():
    return {
        "version": "v1.0.1",
        "hertz": 10.0,
        "name": "empty_config",
        "system_prompt_base": "",
        "system_governance": "",
        "system_prompt_examples": "",
        "agent_inputs": [{"type": "nonexistent_input_type"}],
        "cortex_llm": {"type": "test_llm", "config": {}},
        "simulators": [],
        "agent_actions": [],
    }


@pytest.fixture
def mock_multiple_components_config():
    return {
        "version": "v1.0.1",
        "hertz": 20.0,
        "name": "multiple_components",
        "system_prompt_base": "system prompt base",
        "system_governance": "system governance",
        "system_prompt_examples": "system prompt examples",
        "agent_inputs": [
            {"type": "test_input_1"},
            {"type": "test_input_2"},
        ],
        "cortex_llm": {"type": "test_llm", "config": {"model": "test-model"}},
        "simulators": [{"type": "test_simulator_1"}, {"type": "test_simulator_2"}],
        "agent_actions": [
            {
                "name": "test_action_1",
                "llm_label": "emotion",
                "connector": "test_connector_1",
                "exclude_from_prompt": False,
            },
            {
                "name": "test_action_2",
                "llm_label": "emotion",
                "connector": "test_connector_2",
                "exclude_from_prompt": True,
            },
        ],
    }


def test_load_config(mock_config_data, mock_dependencies):
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config_data))),
        patch(
            "runtime.single_mode.config.load_input",
            return_value=mock_dependencies["input"](),
        ),
        patch(
            "runtime.single_mode.config.load_action",
            return_value=mock_dependencies["action"](),
        ),
        patch(
            "runtime.single_mode.config.load_simulator",
            return_value=mock_dependencies["simulator"](),
        ),
        patch(
            "runtime.single_mode.config.load_llm",
            return_value=mock_dependencies["llm"](),
        ),
    ):
        config = load_config("test_config")

        assert isinstance(config, RuntimeConfig)
        assert config.hertz == mock_config_data["hertz"]
        assert config.name == mock_config_data["name"]
        assert config.system_prompt_base == mock_config_data["system_prompt_base"]
        assert config.system_governance == mock_config_data["system_governance"]
        assert (
            config.system_prompt_examples == mock_config_data["system_prompt_examples"]
        )
        assert config.api_key == mock_config_data["api_key"]
        assert len(config.agent_inputs) == 1
        assert isinstance(config.agent_inputs[0], mock_dependencies["input"])
        assert isinstance(config.cortex_llm, mock_dependencies["llm"])
        assert len(config.simulators) == 1
        assert isinstance(config.simulators[0], mock_dependencies["simulator"])
        assert len(config.agent_actions) == 1
        assert isinstance(config.agent_actions[0], mock_dependencies["action"])


def test_load_empty_config(mock_empty_config_data, mock_dependencies):
    with (
        patch("builtins.open", mock_open(read_data=json.dumps(mock_empty_config_data))),
        patch(
            "runtime.single_mode.config.load_input",
            return_value=mock_dependencies["input"](),
        ),
        patch(
            "runtime.single_mode.config.load_action",
            return_value=mock_dependencies["action"](),
        ),
        patch(
            "runtime.single_mode.config.load_simulator",
            return_value=mock_dependencies["simulator"](),
        ),
        patch(
            "runtime.single_mode.config.load_llm",
            return_value=mock_dependencies["llm"](),
        ),
    ):
        config = load_config("empty_config")

        assert isinstance(config, RuntimeConfig)
        assert config.hertz == mock_empty_config_data["hertz"]
        assert config.name == mock_empty_config_data["name"]
        assert config.system_prompt_base == ""
        assert config.system_governance == ""
        assert config.system_prompt_examples == ""
        assert len(config.agent_inputs) == 1
        assert isinstance(config.agent_inputs[0], mock_dependencies["input"])
        assert isinstance(config.cortex_llm, mock_dependencies["llm"])
        assert len(config.simulators) == 0
        assert len(config.agent_actions) == 0


def test_load_multiple_components(mock_multiple_components_config, mock_dependencies):
    with (
        patch(
            "builtins.open",
            mock_open(read_data=json.dumps(mock_multiple_components_config)),
        ),
        patch(
            "runtime.single_mode.config.load_input",
            return_value=mock_dependencies["input"](),
        ),
        patch(
            "runtime.single_mode.config.load_action",
            return_value=mock_dependencies["action"](),
        ),
        patch(
            "runtime.single_mode.config.load_simulator",
            return_value=mock_dependencies["simulator"](),
        ),
        patch(
            "runtime.single_mode.config.load_llm",
            return_value=mock_dependencies["llm"](),
        ),
    ):
        config = load_config("multiple_components")

        assert isinstance(config, RuntimeConfig)
        assert config.hertz == mock_multiple_components_config["hertz"]
        assert config.name == mock_multiple_components_config["name"]
        assert len(config.agent_inputs) == 2
        assert len(config.simulators) == 2
        assert len(config.agent_actions) == 2


def test_load_config_missing_required_fields():
    invalid_config = {
        "version": "v1.0.1",
        "name": "invalid_config",
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(invalid_config))):
        with pytest.raises(KeyError):
            load_config("invalid_config")


def test_load_config_invalid_version():
    invalid_config = {
        "version": "invalid_version",
        "hertz": 10.0,
        "name": "invalid_version_config",
        "system_prompt_base": "system prompt base",
        "system_governance": "system governance",
        "system_prompt_examples": "system prompt examples",
        "agent_inputs": [],
        "cortex_llm": {"type": "test_llm", "config": {}},
        "simulators": [],
        "agent_actions": [],
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(invalid_config))):
        with pytest.raises(ValueError):
            load_config("invalid_version_config")


def test_load_config_invalid_hertz():
    invalid_config = {
        "version": "v1.0.1",
        "hertz": -1.0,
        "name": "invalid_hertz",
        "system_prompt_base": "system prompt base",
        "system_governance": "system governance",
        "system_prompt_examples": "system prompt examples",
        "agent_inputs": [],
        "cortex_llm": {"type": "test_llm", "config": {}},
        "simulators": [],
        "agent_actions": [],
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(invalid_config))):
        with pytest.raises(ValueError):
            load_config("invalid_config")


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config")


def test_load_config_invalid_json():
    with patch("builtins.open", mock_open(read_data="invalid json5")):
        with pytest.raises(ValueError):
            load_config("invalid_config")


def test_load_config_invalid_component_type():
    invalid_config = {
        "version": "v1.0.1",
        "hertz": 10.0,
        "name": "invalid_component",
        "system_prompt_base": "system prompt base",
        "system_governance": "system governance",
        "system_prompt_examples": "system prompt examples",
        "agent_inputs": [{"type": "nonexistent_input_type"}],
        "cortex_llm": {"type": "test_llm", "config": {}},
        "simulators": [],
        "agent_actions": [],
    }

    with (
        patch("builtins.open", mock_open(read_data=json.dumps(invalid_config))),
        patch("runtime.single_mode.config.load_input", side_effect=ImportError),
    ):
        with pytest.raises(ImportError):
            load_config("invalid_config")


@contextmanager
def temp_env(key: str, value: Optional[str]):
    """Temporarily set environment variable."""
    old_value = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def test_load_config_missing_api_key_warns(caplog, mock_dependencies):
    """Test that missing OM_API_KEY logs warning but doesn't raise."""
    config_data = {
        "version": "v1.0.1",
        "hertz": 10.0,
        "name": "test_missing_key",
        "api_key": "",  # Empty key should trigger env check
        "system_prompt_base": "test",
        "system_governance": "test",
        "system_prompt_examples": "test",
        "agent_inputs": [],
        "cortex_llm": {"type": "test_llm", "config": {}},
        "simulators": [],
        "agent_actions": [],
    }

    with (
        patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        patch(
            "runtime.single_mode.config.load_llm",
            return_value=mock_dependencies["llm"],
        ),
        temp_env("OM_API_KEY", None),
    ):
        config = load_config("test_missing_key")
        assert isinstance(config, RuntimeConfig)
        # Should have logged warning about missing API key
        assert any(
            "API key" in record.message or "OM_API_KEY" in record.message
            for record in caplog.records
        )


def test_load_config_empty_api_key_falls_back_to_env(caplog, mock_dependencies):
    """Test that empty api_key in config falls back to OM_API_KEY env var."""
    caplog.set_level(logging.INFO)

    config_data = {
        "version": "v1.0.1",
        "hertz": 10.0,
        "name": "test_env_key",
        "api_key": "openmind_free",
        "system_prompt_base": "test",
        "system_governance": "test",
        "system_prompt_examples": "test",
        "agent_inputs": [],
        "cortex_llm": {"type": "test_llm", "config": {}},
        "simulators": [],
        "agent_actions": [],
    }

    with (
        patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
        patch(
            "runtime.single_mode.config.load_llm",
            return_value=mock_dependencies["llm"],
        ),
        temp_env("OM_API_KEY", "test_env_api_key_12345"),
    ):
        config = load_config("test_env_key")
        assert isinstance(config, RuntimeConfig)
        # API key should be taken from environment
        assert config.api_key == "test_env_api_key_12345"
        # Should log success message
        assert any(
            "OM_API_KEY" in record.message and "Success" in record.message
            for record in caplog.records
        )
