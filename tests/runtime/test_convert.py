import copy

import pytest

from runtime.converter import (
    ConfigConverter,
    convert_to_multi_mode,
)

SINGLE_MODE_CONFIG = {
    "version": "0.3.0",
    "name": "test_robot",
    "hertz": 2.0,
    "system_prompt_base": "You are a test robot.",
    "system_governance": "Be safe.",
    "system_prompt_examples": "Example 1.",
    "api_key": "test_key",
    "robot_ip": "192.168.1.1",
    "URID": "test-urid",
    "unitree_ethernet": "eth0",
    "cortex_llm": {"type": "TestLLM", "model": "test-model"},
    "agent_inputs": [{"type": "TestInput"}],
    "agent_actions": [{"name": "TestAction"}],
    "simulators": [{"type": "TestSim"}],
    "backgrounds": [{"type": "TestBG"}],
    "action_execution_mode": "parallel",
    "action_dependencies": {"TestAction": ["OtherAction"]},
}

MULTI_MODE_CONFIG = {
    "version": "0.3.0",
    "name": "test_modes",
    "default_mode": "mode_a",
    "allow_manual_switching": True,
    "mode_memory_enabled": False,
    "cortex_llm": {"type": "TestLLM"},
    "system_governance": "Be safe.",
    "modes": {
        "mode_a": {
            "display_name": "Mode A",
            "description": "First mode",
            "hertz": 1.0,
            "system_prompt_base": "Prompt A",
            "agent_inputs": [],
            "agent_actions": [],
            "simulators": [],
            "backgrounds": [],
        }
    },
    "transition_rules": [],
}


class TestIsSingleMode:
    def test_single_mode_detected(self):
        assert ConfigConverter.is_single_mode(SINGLE_MODE_CONFIG) is True

    def test_multi_mode_detected(self):
        assert ConfigConverter.is_single_mode(MULTI_MODE_CONFIG) is False

    def test_partial_keys_treated_as_single(self):
        """Config with 'modes' but no 'default_mode' is still single-mode."""
        partial = {"modes": {}, "name": "partial"}
        assert ConfigConverter.is_single_mode(partial) is True

    def test_empty_config_is_single(self):
        assert ConfigConverter.is_single_mode({}) is True


class TestConvertToMultiMode:
    def test_multi_mode_passthrough(self):
        """Multi-mode configs should be returned unchanged."""
        original = copy.deepcopy(MULTI_MODE_CONFIG)
        result = convert_to_multi_mode(MULTI_MODE_CONFIG)
        assert result == original
        assert result is MULTI_MODE_CONFIG

    def test_single_mode_conversion(self):
        config = copy.deepcopy(SINGLE_MODE_CONFIG)
        result = convert_to_multi_mode(config)

        # Top-level structure
        assert "modes" in result
        assert "default_mode" in result
        assert result["default_mode"] == "test_robot"
        assert "test_robot" in result["modes"]

    def test_global_fields_promoted(self):
        config = copy.deepcopy(SINGLE_MODE_CONFIG)
        result = convert_to_multi_mode(config)

        assert result["api_key"] == "test_key"
        assert result["robot_ip"] == "192.168.1.1"
        assert result["URID"] == "test-urid"
        assert result["unitree_ethernet"] == "eth0"
        assert result["system_governance"] == "Be safe."
        assert result["system_prompt_examples"] == "Example 1."

    def test_mode_fields_nested(self):
        config = copy.deepcopy(SINGLE_MODE_CONFIG)
        result = convert_to_multi_mode(config)
        mode = result["modes"]["test_robot"]

        assert mode["hertz"] == 2.0
        assert mode["system_prompt_base"] == "You are a test robot."
        assert mode["agent_inputs"] == [{"type": "TestInput"}]
        assert mode["agent_actions"] == [{"name": "TestAction"}]
        assert mode["simulators"] == [{"type": "TestSim"}]
        assert mode["backgrounds"] == [{"type": "TestBG"}]
        assert mode["action_execution_mode"] == "parallel"
        assert mode["action_dependencies"] == {"TestAction": ["OtherAction"]}

    def test_cortex_llm_at_both_levels(self):
        config = copy.deepcopy(SINGLE_MODE_CONFIG)
        result = convert_to_multi_mode(config)

        assert result["cortex_llm"] == {"type": "TestLLM", "model": "test-model"}
        assert result["modes"]["test_robot"]["cortex_llm"] == {
            "type": "TestLLM",
            "model": "test-model",
        }

    def test_manual_switching_disabled(self):
        config = copy.deepcopy(SINGLE_MODE_CONFIG)
        result = convert_to_multi_mode(config)

        assert result["allow_manual_switching"] is False
        assert result["mode_memory_enabled"] is False

    def test_transition_rules_empty(self):
        config = copy.deepcopy(SINGLE_MODE_CONFIG)
        result = convert_to_multi_mode(config)

        assert result["transition_rules"] == []

    def test_default_name_fallback(self):
        config = {"version": "0.1.0"}
        result = convert_to_multi_mode(config)

        assert result["default_mode"] == "default"
        assert "default" in result["modes"]

    def test_missing_optional_fields_have_defaults(self):
        config = {
            "name": "minimal",
            "version": "0.1.0",
        }
        result = convert_to_multi_mode(config)
        mode = result["modes"]["minimal"]

        # Global defaults
        assert result["api_key"] == ""
        assert result["robot_ip"] == ""
        assert result["URID"] == "default"
        assert result["unitree_ethernet"] == ""
        assert result["system_governance"] == ""
        assert result["system_prompt_examples"] == ""

        # Mode defaults
        assert mode["hertz"] == 1.0
        assert mode["agent_inputs"] == []
        assert mode["agent_actions"] == []
        assert mode["backgrounds"] == []
        assert mode["simulators"] == []
        assert mode["action_execution_mode"] == "concurrent"
        assert mode["action_dependencies"] == {}


class TestValidateConverted:
    def test_valid_config_passes(self):
        config = copy.deepcopy(SINGLE_MODE_CONFIG)
        result = convert_to_multi_mode(config)
        # Should not raise
        ConfigConverter._validate(result, "test_robot")

    def test_missing_modes_raises(self):
        with pytest.raises(ValueError, match="modes"):
            ConfigConverter._validate({"default_mode": "x"}, "x")

    def test_missing_default_mode_raises(self):
        with pytest.raises(ValueError, match="default_mode"):
            ConfigConverter._validate({"modes": {"x": {}}}, "x")

    def test_mode_name_mismatch_raises(self):
        with pytest.raises(ValueError, match="not in modes"):
            ConfigConverter._validate({"modes": {"a": {}}, "default_mode": "b"}, "b")

    def test_missing_display_name_raises(self):
        with pytest.raises(ValueError, match="display_name"):
            ConfigConverter._validate(
                {"modes": {"x": {"description": "test"}}, "default_mode": "x"},
                "x",
            )

    def test_missing_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            ConfigConverter._validate(
                {"modes": {"x": {"display_name": "X"}}, "default_mode": "x"},
                "x",
            )
