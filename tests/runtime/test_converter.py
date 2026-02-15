"""Test suite for src.runtime.converter module."""

import logging
from unittest.mock import patch

import pytest

from src.runtime.converter import ConfigConverter, convert_to_multi_mode


class TestConfigConverter:
    """Test suite for ConfigConverter class."""

    def test_is_single_mode_with_multi_mode_config(self):
        """Test detection of multi-mode config (should return False)."""
        multi_config = {"modes": {"default": {}}, "default_mode": "default"}
        assert ConfigConverter.is_single_mode(multi_config) is False

    def test_is_single_mode_with_single_mode_config(self):
        """Test detection of single-mode config (should return True)."""
        single_config = {"name": "test", "api_key": "123"}
        assert ConfigConverter.is_single_mode(single_config) is True

    def test_is_single_mode_with_empty_config(self):
        """Test detection of empty config."""
        assert ConfigConverter.is_single_mode({}) is True

    def test_convert_to_multi_mode_returns_unchanged_if_already_multi(self):
        """Test that multi-mode configs are returned unchanged."""
        multi_config = {"modes": {"default": {}}, "default_mode": "default"}
        result = ConfigConverter.convert_to_multi_mode(multi_config)
        assert result is multi_config  # should be same object

    def test_convert_to_multi_mode_basic_conversion(self):
        """Test basic conversion of single-mode config to multi-mode."""
        single = {
            "version": "1.0",
            "name": "test_mode",
            "api_key": "test_key",
            "robot_ip": "192.168.1.1",
            "URID": "test-urid",
            "unitree_ethernet": "eth0",
            "system_governance": "test_gov",
            "system_prompt_examples": "examples",
            "cortex_llm": {"provider": "openai"},
            "hertz": 10,
            "system_prompt_base": "base prompt",
            "agent_inputs": ["input1"],
            "agent_actions": ["action1"],
            "backgrounds": ["bg1"],
            "simulators": ["sim1"],
            "action_execution_mode": "sequential",
            "action_dependencies": {"dep": []},
        }

        with patch.object(logging, "info") as mock_log:
            result = ConfigConverter.convert_to_multi_mode(single)

        # Check global section
        assert result["version"] == "1.0"
        assert result["name"] == "test_mode"
        assert result["default_mode"] == "test_mode"
        assert result["allow_manual_switching"] is False
        assert result["mode_memory_enabled"] is False
        assert result["api_key"] == "test_key"
        assert result["robot_ip"] == "192.168.1.1"
        assert result["URID"] == "test-urid"
        assert result["unitree_ethernet"] == "eth0"
        assert result["system_governance"] == "test_gov"
        assert result["system_prompt_examples"] == "examples"
        assert result["cortex_llm"] == {"provider": "openai"}

        # Check modes section
        assert "modes" in result
        assert "test_mode" in result["modes"]
        mode = result["modes"]["test_mode"]
        assert mode["display_name"] == "test_mode"
        assert mode["description"] == "Converted from single-mode config 'test_mode'"
        assert mode["hertz"] == 10
        assert mode["system_prompt_base"] == "base prompt"
        assert mode["agent_inputs"] == ["input1"]
        assert mode["agent_actions"] == ["action1"]
        assert mode["backgrounds"] == ["bg1"]
        assert mode["simulators"] == ["sim1"]
        assert mode["cortex_llm"] == {"provider": "openai"}
        assert mode["action_execution_mode"] == "sequential"
        assert mode["action_dependencies"] == {"dep": []}

        # Check transition_rules
        assert result["transition_rules"] == []

        mock_log.assert_any_call("Converting single-mode config 'test_mode'")
        mock_log.assert_any_call("Conversion validated: config 'test_mode'")

    def test_convert_to_multi_mode_with_missing_optional_fields(self):
        """Test conversion with minimal config (missing optional fields)."""
        single = {"name": "minimal"}

        result = ConfigConverter.convert_to_multi_mode(single)

        assert result["version"] is None
        assert result["api_key"] == ""
        assert result["robot_ip"] == ""
        assert result["URID"] == "default"
        assert result["unitree_ethernet"] == ""
        assert result["system_governance"] == ""
        assert result["system_prompt_examples"] == ""
        assert result["cortex_llm"] is None

        mode = result["modes"]["minimal"]
        assert mode["hertz"] == 1.0  # default
        assert mode["system_prompt_base"] == ""
        assert mode["agent_inputs"] == []
        assert mode["agent_actions"] == []
        assert mode["backgrounds"] == []
        assert mode["simulators"] == []
        assert mode["cortex_llm"] is None
        assert mode["action_execution_mode"] == "concurrent"  # default
        assert mode["action_dependencies"] == {}

    def test_convert_to_multi_mode_raises_on_missing_default_mode(self):
        """Test that validation catches missing default_mode."""
        with patch.object(ConfigConverter, "_build_global_section", return_value={}):
            with pytest.raises(
                ValueError, match="missing global required field 'default_mode'"
            ):
                ConfigConverter.convert_to_multi_mode({"name": "test"})

    def test_convert_to_multi_mode_function_alias(self):
        """Test that the module-level alias works."""
        single = {"name": "alias_test"}
        result = convert_to_multi_mode(single)
        assert result["default_mode"] == "alias_test"
